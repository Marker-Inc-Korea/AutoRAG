param(
  [Parameter(Mandatory=$true)][string]$SyncRoot,
  [Parameter(Mandatory=$true)][string]$SourceRoot,
  [string]$ProviderName = 'AutoRAG Synthetic Cloud Files'
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$source = @'
using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public static class AutoragSyntheticCloudFiles
{
    const string Dll = "cldapi.dll";
    const uint CF_CALLBACK_TYPE_FETCH_DATA = 0;
    const uint CF_CALLBACK_TYPE_NONE = 0xffffffff;
    const uint CF_OPERATION_TYPE_TRANSFER_DATA = 0;
    const uint CF_HYDRATION_POLICY_FULL = 2;
    const ushort CF_HYDRATION_POLICY_MODIFIER_AUTO_DEHYDRATION_ALLOWED = 4;
    const uint CF_POPULATION_POLICY_ALWAYS_FULL = 3;
    const uint CF_PLACEHOLDER_CREATE_FLAG_MARK_IN_SYNC = 2;
    const uint CF_CONNECT_FLAG_REQUIRE_FULL_FILE_PATH = 4;
    const uint CF_CONNECT_FLAG_REQUIRE_PROCESS_INFO = 2;
    const int STATUS_SUCCESS = 0;
    const uint FILE_ATTRIBUTE_NORMAL = 0x80;

    [StructLayout(LayoutKind.Sequential)] public struct UShortPair { public ushort Primary; public ushort Modifier; }
    [StructLayout(LayoutKind.Sequential)] public struct Policies {
        public uint StructSize; public UShortPair Hydration; public UShortPair Population;
        public uint InSync; public uint HardLink; public uint PlaceholderManagement;
    }
    [StructLayout(LayoutKind.Sequential)] public struct Registration {
        public uint StructSize; public IntPtr ProviderName; public IntPtr ProviderVersion;
        public IntPtr SyncRootIdentity; public uint SyncRootIdentityLength;
        public IntPtr FileIdentity; public uint FileIdentityLength;
    }
    [StructLayout(LayoutKind.Sequential)] public struct FileBasicInfo {
        public long CreationTime; public long LastAccessTime; public long LastWriteTime; public long ChangeTime;
        public uint FileAttributes; public uint Reserved;
    }
    [StructLayout(LayoutKind.Sequential)] public struct FsMetadata { public FileBasicInfo BasicInfo; public long FileSize; }
    [StructLayout(LayoutKind.Sequential)] public struct PlaceholderCreateInfo {
        public IntPtr RelativeFileName; public FsMetadata FsMetadata; public IntPtr FileIdentity;
        public uint FileIdentityLength; public uint Flags; public int Result; public long CreateUsn;
    }
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] public delegate void Callback(IntPtr info, IntPtr parameters);
    [StructLayout(LayoutKind.Sequential)] public struct CallbackRegistration { public uint Type; public Callback Callback; }
    [StructLayout(LayoutKind.Sequential)] public struct ConnectionKey { public long Internal; }
    [StructLayout(LayoutKind.Sequential)] public struct CallbackInfo {
        public uint StructSize; public ConnectionKey ConnectionKey; public IntPtr CallbackContext;
        public IntPtr VolumeGuidName; public IntPtr VolumeDosName; public uint VolumeSerialNumber;
        public long SyncRootFileId; public IntPtr SyncRootIdentity; public uint SyncRootIdentityLength;
        public long FileId; public long FileSize; public IntPtr FileIdentity; public uint FileIdentityLength;
        public IntPtr NormalizedPath; public long TransferKey; public byte PriorityHint; public byte Pad1; public byte Pad2; public byte Pad3; public byte Pad4; public byte Pad5; public byte Pad6; public byte Pad7;
        public IntPtr CorrelationVector; public IntPtr ProcessInfo; public long RequestKey;
    }
    [StructLayout(LayoutKind.Sequential)] public struct OperationInfo {
        public uint StructSize; public uint Type; public ConnectionKey ConnectionKey; public long TransferKey;
        public IntPtr CorrelationVector; public IntPtr SyncStatus; public long RequestKey;
    }

[DllImport(Dll, CharSet=CharSet.Unicode)] static extern int CfCreatePlaceholders(string baseDirectory, IntPtr placeholderArray, uint placeholderCount, uint createFlags, out uint entriesProcessed);
    [StructLayout(LayoutKind.Explicit)] public struct OperationParameters {
        [FieldOffset(0)] public uint ParamSize;
        [FieldOffset(8)] public uint TransferFlags;
        [FieldOffset(12)] public int CompletionStatus;
        [FieldOffset(16)] public IntPtr Buffer;
        [FieldOffset(24)] public long Offset;
        [FieldOffset(32)] public long Length;
    }
[DllImport(Dll)] static extern int CfExecute(ref OperationInfo info, ref OperationParameters parameters);
    [DllImport(Dll, CharSet=CharSet.Unicode)] static extern int CfRegisterSyncRoot(string path, ref Registration registration, ref Policies policies, uint flags);
    [DllImport(Dll, CharSet=CharSet.Unicode)] static extern int CfUnregisterSyncRoot(string path);
    [DllImport(Dll, CharSet=CharSet.Unicode)] static extern int CfConnectSyncRoot(string path, [In] CallbackRegistration[] callbacks, IntPtr context, uint flags, out ConnectionKey key);
    [DllImport(Dll)] static extern int CfDisconnectSyncRoot(ConnectionKey key);

    static string sourceRoot;
    static string syncRootPath;
    static Callback callback;
    static ConnectionKey connection;

    static long FileTime(string path) { return File.GetLastWriteTimeUtc(path).ToFileTimeUtc(); }
    static string Relative(string path) {
        string root = sourceRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return path.Substring(root.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar).Replace('/', '\\');
    }

    static void FetchData(IntPtr infoPtr, IntPtr parametersPtr)
    {
        try {
            var info = Marshal.PtrToStructure<CallbackInfo>(infoPtr);
            long offset = Marshal.ReadInt64(parametersPtr, 16);
            long length = Marshal.ReadInt64(parametersPtr, 24);
            
            string normalized = Marshal.PtrToStringUni(info.NormalizedPath) ?? "";
            string rel;
            if (info.NormalizedPath != IntPtr.Zero && normalized.Length > 0) {
                rel = normalized;
                if (rel.StartsWith(syncRootPath, StringComparison.OrdinalIgnoreCase)) rel = rel.Substring(syncRootPath.Length);
            } else if (info.FileIdentity != IntPtr.Zero && info.FileIdentityLength > 0) {
                var idBytes = new byte[info.FileIdentityLength];
                Marshal.Copy(info.FileIdentity, idBytes, 0, (int)info.FileIdentityLength);
                rel = Encoding.UTF8.GetString(idBytes);
            } else {
                throw new IOException("callback carried neither a path nor a file identity");
            }
            rel = rel.TrimStart('\\');
            string sourcePath = Path.Combine(sourceRoot, rel.Replace('\\', Path.DirectorySeparatorChar));
            if (!File.Exists(sourcePath)) sourcePath = Path.Combine(sourceRoot, Path.GetFileName(rel));
            byte[] data = File.ReadAllBytes(sourcePath);
            if (offset < 0 || offset > data.LongLength) throw new IOException("invalid offset");
            int take = (int)Math.Min(length, data.LongLength - offset);
            var buffer = new byte[take];
            Buffer.BlockCopy(data, (int)offset, buffer, 0, take);
            var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            try {
                var opInfo = new OperationInfo { StructSize=(uint)Marshal.SizeOf<OperationInfo>(), Type=CF_OPERATION_TYPE_TRANSFER_DATA, ConnectionKey=info.ConnectionKey, TransferKey=info.TransferKey, CorrelationVector=info.CorrelationVector, RequestKey=info.RequestKey };
                var opParams = new OperationParameters { ParamSize=(uint)Marshal.SizeOf<OperationParameters>(), TransferFlags=0, CompletionStatus=STATUS_SUCCESS, Buffer=handle.AddrOfPinnedObject(), Offset=offset, Length=take };
                int hr = CfExecute(ref opInfo, ref opParams);
                if (hr != 0) Console.Error.WriteLine("FETCH_DATA_EXEC=0x" + ((uint)hr).ToString("X8"));
            } finally { handle.Free(); }
        } catch (Exception ex) { Console.Error.WriteLine("FETCH_DATA_ERROR=" + ex.Message); }
    }

    public static string Start(string syncRootArg, string sourceRootArg, string providerName)
    {
        sourceRoot = Path.GetFullPath(sourceRootArg);
        syncRootPath = Path.GetFullPath(syncRootArg).TrimEnd(Path.DirectorySeparatorChar);
        Directory.CreateDirectory(syncRootArg);
        var providerPtr = Marshal.StringToHGlobalUni(providerName);
        var versionPtr = Marshal.StringToHGlobalUni("1.0");
        try {
            var reg = new Registration { StructSize=(uint)Marshal.SizeOf<Registration>(), ProviderName=providerPtr, ProviderVersion=versionPtr };
            var policy = new Policies {
                StructSize=(uint)Marshal.SizeOf<Policies>(),
                Hydration=new UShortPair { Primary=(ushort)CF_HYDRATION_POLICY_FULL, Modifier=(ushort)CF_HYDRATION_POLICY_MODIFIER_AUTO_DEHYDRATION_ALLOWED },
                Population=new UShortPair { Primary=(ushort)CF_POPULATION_POLICY_ALWAYS_FULL, Modifier=0 },
                InSync=0, HardLink=0, PlaceholderManagement=1
            };
            int hr = CfRegisterSyncRoot(syncRootArg, ref reg, ref policy, 0);
            if (hr != 0) throw new Exception("CfRegisterSyncRoot=0x" + ((uint)hr).ToString("X8"));
            callback = FetchData;
            var callbacks = new[] { new CallbackRegistration { Type=CF_CALLBACK_TYPE_FETCH_DATA, Callback=callback }, new CallbackRegistration { Type=CF_CALLBACK_TYPE_NONE, Callback=null } };
            hr = CfConnectSyncRoot(syncRootArg, callbacks, IntPtr.Zero, CF_CONNECT_FLAG_REQUIRE_FULL_FILE_PATH | CF_CONNECT_FLAG_REQUIRE_PROCESS_INFO, out connection);
            if (hr != 0) { CfUnregisterSyncRoot(syncRootArg); throw new Exception("CfConnectSyncRoot=0x" + ((uint)hr).ToString("X8")); }
            CreateSourcePlaceholders(syncRootArg);
            return "READY";
        } finally { Marshal.FreeHGlobal(providerPtr); Marshal.FreeHGlobal(versionPtr); }
    }

    static void CreateSourcePlaceholders(string syncRoot)
    {
        var files = Directory.GetFiles(sourceRoot, "*", SearchOption.AllDirectories);
        foreach (string sourceFile in files) {
            string rel = Relative(sourceFile);
            byte[] identityBytes = Encoding.UTF8.GetBytes(rel);
            IntPtr name = Marshal.StringToHGlobalUni(rel);
            IntPtr identity = Marshal.AllocHGlobal(identityBytes.Length);
            Marshal.Copy(identityBytes, 0, identity, identityBytes.Length);
            try {
                var metadata = new FsMetadata {
                    BasicInfo = new FileBasicInfo {
                        CreationTime = FileTime(sourceFile), LastAccessTime = FileTime(sourceFile),
                        LastWriteTime = FileTime(sourceFile), ChangeTime = FileTime(sourceFile),
                        FileAttributes = FILE_ATTRIBUTE_NORMAL,
                    },
                    FileSize = new FileInfo(sourceFile).Length,
                };
                var info = new PlaceholderCreateInfo {
                    RelativeFileName = name, FsMetadata = metadata,
                    FileIdentity = identity, FileIdentityLength = (uint)identityBytes.Length,
                    Flags = CF_PLACEHOLDER_CREATE_FLAG_MARK_IN_SYNC,
                };
                IntPtr array = Marshal.AllocHGlobal(Marshal.SizeOf<PlaceholderCreateInfo>());
                try {
                    Marshal.StructureToPtr(info, array, false);
                    uint processed;
                    int hr = CfCreatePlaceholders(syncRoot, array, 1, 0, out processed);
                    var result = Marshal.PtrToStructure<PlaceholderCreateInfo>(array);
                    if (hr != 0 || result.Result != 0) throw new Exception("CfCreatePlaceholders=0x" + ((uint)(hr != 0 ? hr : result.Result)).ToString("X8"));
                } finally { Marshal.FreeHGlobal(array); }
            } finally { Marshal.FreeHGlobal(name); Marshal.FreeHGlobal(identity); }
        }
    }

    public static void Stop(string syncRoot)
    {
        try { CfDisconnectSyncRoot(connection); } finally { CfUnregisterSyncRoot(syncRoot); }
    }
}
'@
Add-Type -TypeDefinition $source -Language CSharp
try {
  Write-Output ("PROVIDER=" + [AutoragSyntheticCloudFiles]::Start($SyncRoot, $SourceRoot, $ProviderName))
  Write-Output ("ROOT=" + [IO.Path]::GetFullPath($SyncRoot))
  [Console]::Out.Flush()
  while (($line = [Console]::In.ReadLine()) -ne $null) {
    if ($line -eq 'stop') { break }
  }
} finally {
  try { [AutoragSyntheticCloudFiles]::Stop($SyncRoot) } catch { Write-Error $_ }
}
