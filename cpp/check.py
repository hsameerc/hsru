import os
import importlib.util
import sys

# import zipfile
#
# with zipfile.ZipFile("dist/hsru_cuda_kernel-0.4.0-cp311-cp311-win_amd64.whl", "r") as whl:
#     whl.extractall("hsru_extracted")

def find_hsru_pyd():
    import hsru_cuda_kernel
    path = hsru_cuda_kernel.__file__
    print(f"✅ Found hsru_cuda_kernel at: {path}")
    return path

def check_import():
    try:
        import hsru_cuda_kernel
        print("✅ Import succeeded.")
    except ImportError as e:
        print("❌ ImportError:", e)
        print("🔍 Likely missing a DLL dependency.")
        print("💡 Try using Dependencies.exe or dumpbin to inspect the .pyd file.")

def main():
    print("🔎 Checking hsru_cuda_kernel installation...")
    try:
        pyd_path = find_hsru_pyd()
        if os.path.exists(pyd_path):
            print("📦 .pyd file exists.")
        else:
            print("❌ .pyd file not found.")
        check_import()
    except ModuleNotFoundError:
        print("❌ hsru_cuda_kernel not installed in current environment.")
        print("💡 Try: pip install hsru_cuda_kernel")

if __name__ == "__main__":
    main()
