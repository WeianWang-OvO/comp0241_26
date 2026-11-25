"""Names of the cameras, by the index cv2.VideoCapture uses (macOS and Linux; empty elsewhere).

OpenCV itself only knows numbers. This asks the operating system instead, without opening any camera:
  macOS  AVFoundation, through ctypes (no extra package); OpenCV numbers the cameras in order of their uniqueID
  Linux  /sys/class/video4linux/videoN/name, where N is the index, and the USB port from .../videoN/device
"""
import ctypes
import ctypes.util
import glob
import os
import re
import sys


def _mac_devices():
    """[(uniqueID, name)] of the macOS video devices, through the Objective-C runtime."""
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
    ctypes.cdll.LoadLibrary('/System/Library/Frameworks/AVFoundation.framework/AVFoundation')
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value

    def send(restype, *argtypes):  # objc_msgSend with the right C signature
        return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(address)

    def text(obj, selector):
        value = send(ctypes.c_char_p)(send(ctypes.c_void_p)(obj, objc.sel_registerName(selector)),
                                      objc.sel_registerName(b'UTF8String'))
        return value.decode('utf-8', 'replace') if value else ''

    nsstring = objc.objc_getClass(b'NSString')
    device_class = objc.objc_getClass(b'AVCaptureDevice')
    devices = []
    for media in (b'vide', b'muxx'):  # AVMediaTypeVideo and AVMediaTypeMuxed, as in OpenCV
        media_type = send(ctypes.c_void_p, ctypes.c_char_p)(nsstring, objc.sel_registerName(b'stringWithUTF8String:'), media)
        array = send(ctypes.c_void_p, ctypes.c_void_p)(device_class, objc.sel_registerName(b'devicesWithMediaType:'),
                                                       media_type)
        for i in range(send(ctypes.c_ulong)(array, objc.sel_registerName(b'count'))):
            device = send(ctypes.c_void_p, ctypes.c_ulong)(array, objc.sel_registerName(b'objectAtIndex:'), i)
            devices.append((text(device, b'uniqueID'), text(device, b'localizedName')))
    return devices


def linux_names(sysfs='/sys'):
    """{index: name} from sysfs, with the USB port (e.g. 'USB 1-2') to tell two identical webcams apart."""
    names = {}
    for path in glob.glob(os.path.join(sysfs, 'class/video4linux/video*/name')):
        index = int(re.fullmatch(r'video(\d+)', os.path.basename(os.path.dirname(path))).group(1))  # .../videoN/name
        with open(path) as f:
            name = f.read().strip()
        # .../videoN/device links to the USB interface, e.g. .../usb1/1-2/1-2:1.0: its port is 1-2
        port = re.fullmatch(r'(\d+-[\d.]+):[\d.]+', os.path.basename(os.path.realpath(os.path.join(os.path.dirname(path), 'device'))))
        names[index] = f"{name} (USB {port.group(1)})" if port else name
    return dict(sorted(names.items()))


def camera_names():
    """{index: name}. On macOS, USB cameras get their USB location added, to tell two identical webcams apart."""
    try:
        if sys.platform == 'darwin':
            devices = sorted(_mac_devices())  # OpenCV sorts by uniqueID before numbering (OpenCV 4.x)
            names = {}
            for index, (uid, name) in enumerate(devices):
                usb = re.fullmatch(r'0x([0-9a-fA-F]{8})[0-9a-fA-F]+', uid)  # USB cameras: location ID, vendor, product
                names[index] = f"{name} (USB {usb.group(1)})" if usb else name
            return names
        if sys.platform.startswith('linux'):
            return linux_names()
    except Exception:  # names are only a help: never stop the program for them
        pass
    return {}


BUILT_IN = re.compile(r'facetime|macbook|built-?in|integrated|iphone|ipad|virtual|obs|snap', re.I)


def likely_pair(names):
    """Two indices that are probably the stereo pair: two cameras of the same model, else two external cameras."""
    by_model = {}
    for index, name in names.items():
        if not name:  # no name: nothing to compare
            continue
        by_model.setdefault(re.sub(r' \(USB [0-9a-fA-F.-]+\)$', '', name), []).append(index)
    for model, indices in by_model.items():
        if len(indices) >= 2 and not BUILT_IN.search(model):
            return indices[0], indices[1]
    external = [i for i, name in names.items() if name and not BUILT_IN.search(name)]
    return (external[0], external[1]) if len(external) >= 2 else None


def camera_label(index, names):
    """'1: Arducam OV9782 USB Camera (USB 21300000)', or just '1' when there is no name."""
    return f"{index}: {names[index]}" if index in names else str(index)


def record_cameras(folder, left, right):
    """Note in <folder>/cameras.txt which camera took the left and which the right photos."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'cameras.txt'), 'w', encoding='utf-8') as f:
        f.write(f"left: {left}\nright: {right}\n")


def recorded_cameras(folder):
    """(left, right) labels from <folder>/cameras.txt, or None if there is no such note."""
    try:
        with open(os.path.join(folder, 'cameras.txt'), encoding='utf-8') as f:
            found = dict(line.split(':', 1) for line in f.read().splitlines() if ':' in line)
        return found['left'].strip(), found['right'].strip()
    except (OSError, KeyError, ValueError):
        return None


def compare_cameras(recorded, current):
    """'same', 'swapped' or 'different'. Names (with their USB location) survive renumbering; indices do not."""
    identity = lambda label: label.split(':', 1)[1].strip() if ':' in label else label.strip()
    a, b = tuple(map(identity, recorded)), tuple(map(identity, current))
    if a == b:
        return 'same'
    return 'swapped' if a == b[::-1] else 'different'


if __name__ == '__main__':
    for index, name in camera_names().items():
        print(f"{index}: {name}")
    print("likely stereo pair:", likely_pair(camera_names()))
