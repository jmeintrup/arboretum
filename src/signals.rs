use core::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

static SIGINT: AtomicBool = AtomicBool::new(false);

pub fn received_ctrl_c() -> bool {
    SIGINT.load(Ordering::SeqCst)
}

pub fn initialize() {
    ctrlc::set_handler(|| {
        SIGINT.store(true, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");
}
