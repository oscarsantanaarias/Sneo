import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

const folderDialog = `
using System;
using System.Runtime.InteropServices;

[ComImport, Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")]
class SneoOpenDialog {}

[ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface SneoShellItem {
    void BindToHandler(IntPtr bindContext, ref Guid handler, ref Guid kind, out IntPtr result);
    void GetParent(out SneoShellItem parent);
    void GetDisplayName(uint form, [MarshalAs(UnmanagedType.LPWStr)] out string name);
    void GetAttributes(uint mask, out uint attributes);
    void Compare(SneoShellItem other, uint hint, out int order);
}

[ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface SneoFileDialog {
    [PreserveSig] int Show(IntPtr owner);
    void SetFileTypes(uint count, IntPtr filters);
    void SetFileTypeIndex(uint index);
    void GetFileTypeIndex(out uint index);
    void Advise(IntPtr events, out uint cookie);
    void Unadvise(uint cookie);
    void SetOptions(uint options);
    void GetOptions(out uint options);
    void SetDefaultFolder(SneoShellItem item);
    void SetFolder(SneoShellItem item);
    void GetFolder(out SneoShellItem item);
    void GetCurrentSelection(out SneoShellItem item);
    void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
    void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
    void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
    void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
    void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
    void GetResult(out SneoShellItem item);
    void AddPlace(SneoShellItem item, int place);
    void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string extension);
    void Close(int result);
    void SetClientGuid(ref Guid guid);
    void ClearClientData();
    void SetFilter(IntPtr filter);
}

public static class SneoFolderPicker {
    [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
    static extern void SHCreateItemFromParsingName(string path, IntPtr bindContext, ref Guid kind, out SneoShellItem item);

    [DllImport("user32.dll")]
    static extern IntPtr GetForegroundWindow();

    public static string Pick(string title, string start, string button) {
        SneoFileDialog dialog = (SneoFileDialog)new SneoOpenDialog();
        dialog.SetOptions(0x20 | 0x40 | 0x800);
        dialog.SetTitle(title);
        dialog.SetOkButtonLabel(button);
        if(!string.IsNullOrEmpty(start) && System.IO.Directory.Exists(start)){
            try {
                Guid kind = typeof(SneoShellItem).GUID;
                SneoShellItem item;
                SHCreateItemFromParsingName(start, IntPtr.Zero, ref kind, out item);
                dialog.SetFolder(item);
            } catch {}
        }
        if(dialog.Show(GetForegroundWindow()) != 0) return "";
        SneoShellItem picked;
        dialog.GetResult(out picked);
        string path;
        picked.GetDisplayName(0x80058000, out path);
        return path;
    }
}
`;

// los textos van por variables de entorno, nunca dentro del script, asi ninguna comilla puede cerrar el texto
export const pickFolder = async ({title = 'Select a folder', start = '', button = 'Select folder'} = {}) => {
  const script = `$ProgressPreference = 'SilentlyContinue'\n[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\nAdd-Type -TypeDefinition @'${folderDialog}'@\n[SneoFolderPicker]::Pick($env:SNEO_PICK_TITLE, $env:SNEO_PICK_START, $env:SNEO_PICK_BUTTON)`;
  const result = await execFileAsync('powershell.exe', ['-NoProfile', '-STA', '-EncodedCommand', Buffer.from(script, 'utf16le').toString('base64')], {
    timeout: 300000,
    windowsHide: false,
    maxBuffer: 1024 * 100,
    env: {...process.env, SNEO_PICK_TITLE: String(title || ''), SNEO_PICK_START: String(start || ''), SNEO_PICK_BUTTON: String(button || '')}
  });
  return String(result.stdout || '').trim();
};
