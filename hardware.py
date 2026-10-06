"""Read-only NVIDIA inventory, metrics, and worker-local GPU routing."""
import csv
import ctypes
import os
import subprocess
import uuid

HIDDEN=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
FIELDS='index,uuid,name,memory.total,memory.used,utilization.gpu,utilization.encoder,utilization.decoder'


def number(text):
    try:
        return float(text.strip())
    except ValueError:
        return None


def query_gpus():
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu='+FIELDS,'--format=csv,noheader,nounits'],
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=3,creationflags=HIDDEN)
    except (OSError,subprocess.TimeoutExpired):
        return []
    if result.returncode:
        return []
    devices=[]
    for row in csv.reader(result.stdout.splitlines(),skipinitialspace=True):
        if len(row)!=8:
            continue
        try:
            index=int(row[0])
        except ValueError:
            continue
        devices.append({'index':index,'uuid':row[1].strip(),'name':row[2].strip(),
            'memory_total':number(row[3]),'memory_used':number(row[4]),'utilization':number(row[5]),
            'encoder_utilization':number(row[6]),'decoder_utilization':number(row[7])})
    return devices


def choose_gpu(devices,requested='auto'):
    if requested=='cpu':
        return None
    if requested=='auto':
        if not devices:
            raise RuntimeError('未检测到可用 NVIDIA 显卡，请检查驱动或手动选择 CPU。')
        return next((g for g in devices if '4070' in g['name']),max(devices,key=lambda g:g.get('memory_total') or 0))
    selected=next((g for g in devices if requested in (g['uuid'],str(g['index']))),None)
    if selected is None:
        raise RuntimeError('所选显卡已经不可用，请刷新设备后重新选择。')
    return selected


def route_gpu(selected):
    # Configure before any CUDA import. Each worker sees its selected GPU as CUDA:0.
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=selected['uuid'] if selected else ''


def cuda_driver_devices():
    # FFmpeg/PyAV use the driver API, which does not filter by CUDA_VISIBLE_DEVICES.
    driver=ctypes.WinDLL('nvcuda.dll') if os.name=='nt' else ctypes.CDLL('libcuda.so.1')
    def checked(code):
        if code:
            raise RuntimeError(f'无法核对显卡驱动设备（CUDA 错误 {code}），请检查驱动或选择 CPU。')
    checked(driver.cuInit(0))
    count=ctypes.c_int()
    checked(driver.cuDeviceGetCount(ctypes.byref(count)))
    devices=[]
    for index in range(count.value):
        device=ctypes.c_int()
        checked(driver.cuDeviceGet(ctypes.byref(device),index))
        identifier=(ctypes.c_ubyte*16)()
        checked(driver.cuDeviceGetUuid(ctypes.byref(identifier),device))
        name=ctypes.create_string_buffer(256)
        checked(driver.cuDeviceGetName(name,256,device))
        devices.append({'index':index,'uuid':'GPU-'+str(uuid.UUID(bytes=bytes(identifier))),
            'name':name.value.decode('utf-8',errors='replace')})
    return devices


def cuda_driver_index(selected):
    try:
        devices=cuda_driver_devices()
    except OSError as exc:
        raise RuntimeError('无法加载 NVIDIA 显卡驱动，请检查驱动或选择 CPU。') from exc
    actual=next((device for device in devices if device['uuid']==selected['uuid']),None)
    if actual is None:
        raise RuntimeError('显卡驱动没有找到所选设备，已停止处理，请重新选择设备。')
    return actual['index']


def device_event(cfg,gpu=False,loading=False):
    selected=cfg.get('selected_gpu')
    if not gpu:
        return {'active_device_type':'cpu','active_device_name':'CPU','active_gpu_uuid':None}
    if not selected:
        return {'active_device_type':'unknown','active_device_name':'显卡待核验','active_gpu_uuid':None}
    return {'active_device_type':'loading' if loading else 'gpu',
        'active_device_name':selected['name'],'active_gpu_uuid':selected['uuid']}


def windows_display_devices():
    """Present display-class devices; SetupAPI avoids a slow PowerShell/WMI startup."""
    if os.name!='nt': return []
    class GUID(ctypes.Structure):
        _fields_=[('Data1',ctypes.c_uint32),('Data2',ctypes.c_uint16),('Data3',ctypes.c_uint16),('Data4',ctypes.c_ubyte*8)]
    class DeviceInfo(ctypes.Structure):
        _fields_=[('cbSize',ctypes.c_uint32),('ClassGuid',GUID),('DevInst',ctypes.c_uint32),('Reserved',ctypes.c_size_t)]
    api=ctypes.WinDLL('setupapi.dll',use_last_error=True)
    api.SetupDiGetClassDevsW.argtypes=[ctypes.POINTER(GUID),ctypes.c_wchar_p,ctypes.c_void_p,ctypes.c_uint32]
    api.SetupDiGetClassDevsW.restype=ctypes.c_void_p
    api.SetupDiEnumDeviceInfo.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.POINTER(DeviceInfo)]
    api.SetupDiGetDeviceRegistryPropertyW.argtypes=[ctypes.c_void_p,ctypes.POINTER(DeviceInfo),ctypes.c_uint32,
        ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_void_p]
    api.SetupDiDestroyDeviceInfoList.argtypes=[ctypes.c_void_p]
    guid=GUID(0x4d36e968,0xe325,0x11ce,(ctypes.c_ubyte*8)(0xbf,0xc1,0x08,0x00,0x2b,0xe1,0x03,0x18))
    handle=api.SetupDiGetClassDevsW(ctypes.byref(guid),None,None,2)
    if handle==ctypes.c_void_p(-1).value: raise OSError(ctypes.get_last_error(),'显示设备查询失败')
    rows=[]
    try:
        for index in range(256):
            device=DeviceInfo();device.cbSize=ctypes.sizeof(device)
            if not api.SetupDiEnumDeviceInfo(handle,index,ctypes.byref(device)): break
            def read(property_id):
                buffer=ctypes.create_unicode_buffer(2048)
                if api.SetupDiGetDeviceRegistryPropertyW(handle,ctypes.byref(device),property_id,None,
                        buffer,ctypes.sizeof(buffer),None): return buffer.value
                return ''
            rows.append({'Name':read(12) or read(0),'PNPDeviceID':read(1)})
    finally:
        api.SetupDiDestroyDeviceInfoList(handle)
    return rows


def query_display_adapters(cuda_devices):
    adapters=[{'name':g['name'],'supported':True,'note':'可选择 NVIDIA CUDA；模型启动时核验'} for g in cuda_devices]
    try: rows=windows_display_devices()
    except OSError: return adapters
    for row in rows:
        name=row.get('Name') or '未命名显示适配器'
        if any(a['name'].casefold()==name.casefold() for a in adapters): continue
        virtual=not str(row.get('PNPDeviceID','')).upper().startswith('PCI\\')
        adapters.append({'name':name,'supported':False,
            'note':'显示/虚拟适配器，不能用于当前模型' if virtual else '已识别；当前模型加速仅支持 NVIDIA CUDA，可使用 CPU'})
    return adapters


def query_resources():
    """Whole-machine CPU/RAM plus this app's process tree working sets."""
    try:
        import psutil
    except ImportError: return {}
    try:
        memory=psutil.virtual_memory()
        parent=psutil.Process()
        processes=[parent]+parent.children(recursive=True)
        working_set=0
        for process in processes:
            try: working_set+=process.memory_info().rss
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        return {'cpu_percent':psutil.cpu_percent(interval=None),
            'ram_used':memory.total-memory.available,'ram_total':memory.total,'ram_percent':memory.percent,
            'app_working_set':working_set,'processes':len(processes)}
    except (psutil.Error,OSError): return {}
