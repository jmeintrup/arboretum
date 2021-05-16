use core::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::SystemTime;

static SIGINT: AtomicBool = AtomicBool::new(false);
static TIMEOUT: AtomicBool = AtomicBool::new(false);

pub fn received_ctrl_c() -> bool {
    SIGINT.load(Ordering::SeqCst)
}

pub fn initialize() {
    ctrlc::set_handler(|| {
        SIGINT.store(true, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");
}

pub fn initialize_timeout(start_time: SystemTime, max_duration: u64) {
    thread::Builder::new()
        .name("timout".into())
        .spawn(move || loop {
            if start_time.elapsed().expect("failed to obtain elapsed time").as_secs() > max_duration {
                TIMEOUT.store(true, Ordering::SeqCst);
            }
        })
        .expect("failed to spawn thread");
}

pub fn timeout() -> bool {
    TIMEOUT.load(Ordering::SeqCst)
}