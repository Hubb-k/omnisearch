use std::ptr::null_mut;
use std::sync::OnceLock;
use std::sync::Arc;
use windows::core::PCWSTR;
use windows::Win32::Foundation::{HANDLE, HGLOBAL, HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::DataExchange::{
    AddClipboardFormatListener, CloseClipboard, GetClipboardData, OpenClipboard,
};
use windows::Win32::System::Memory::{GlobalLock, GlobalSize, GlobalUnlock};
use windows::Win32::System::Threading::{GetCurrentProcess, SetPriorityClass, IDLE_PRIORITY_CLASS};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetMessageW, RegisterClassW,
    WINDOW_EX_STYLE, WINDOW_STYLE, WNDCLASSW, WM_CLIPBOARDUPDATE, MSG,
};

const CF_UNICODETEXT: u32 = 13;

static CALLBACK: OnceLock<Arc<dyn Fn(String) + Send + Sync>> = OnceLock::new();

unsafe extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if msg == WM_CLIPBOARDUPDATE {
        if OpenClipboard(hwnd).is_ok() {
            if let Ok(handle) = GetClipboardData(CF_UNICODETEXT) {
                if !handle.is_invalid() && handle != HANDLE::default() {
                    let hglobal = HGLOBAL(handle.0 as *mut _);
                    let locked_ptr = GlobalLock(hglobal);
                    if !locked_ptr.is_null() {
                        let text_ptr = locked_ptr as *const u16;
                        let byte_size = GlobalSize(hglobal);
                        let char_count = (byte_size / 2).saturating_sub(1) as usize;

                        let mut actual_len = 0;
                        while actual_len < char_count && *text_ptr.add(actual_len) != 0 {
                            actual_len += 1;
                        }

                        let slice = std::slice::from_raw_parts(text_ptr, actual_len);
                        if let Ok(clean_text) = String::from_utf16(slice) {
                            if !clean_text.trim().is_empty() {
                                if let Some(cb) = CALLBACK.get() {
                                    cb(clean_text);
                                }
                            }
                        }
                        let _ = GlobalUnlock(hglobal);
                    }
                }
            }
            let _ = CloseClipboard();
        }
        return LRESULT(0);
    }
    DefWindowProcW(hwnd, msg, wparam, lparam)
}

pub fn start_listener<F>(callback: F) -> Result<(), windows::core::Error>
where
    F: Fn(String) + Send + Sync + 'static,
{
    let _ = CALLBACK.set(Arc::new(callback));

    unsafe {
        SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS)?;

        let class_name: Vec<u16> = "NetIntelListener\0".encode_utf16().collect();
        let wnd_class = WNDCLASSW {
            lpfnWndProc: Some(wnd_proc),
            hInstance: windows::Win32::System::LibraryLoader::GetModuleHandleW(None)?.into(),
            lpszClassName: PCWSTR(class_name.as_ptr()),
            ..Default::default()
        };
        RegisterClassW(&wnd_class);

        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            PCWSTR(class_name.as_ptr()),
            PCWSTR(null_mut()),
            WINDOW_STYLE(0),
            0, 0, 0, 0,
            HWND(0),
            None, None, None,
        );

        AddClipboardFormatListener(hwnd)?;
        println!("[Синапс] Слушатель активен.");

        let mut msg = MSG::default();
        while GetMessageW(&mut msg, HWND(0), 0, 0).as_bool() {
            DispatchMessageW(&msg);
        }
    }
    Ok(())
}