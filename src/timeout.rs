use core::sync::atomic::AtomicBool;
use core::time;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::SystemTime;

static TIMEOUT: AtomicBool = AtomicBool::new(false);

pub fn initialize_timeout(max_duration: u64) {
    let start_time = SystemTime::now();
    thread::Builder::new()
        .name("timout".into())
        .spawn(move || loop {
            thread::sleep(time::Duration::from_millis(10));
            if start_time
                .elapsed()
                .expect("failed to obtain elapsed time")
                .as_secs()
                > max_duration
            {
                TIMEOUT.store(true, Ordering::SeqCst);
            }
        })
        .expect("failed to spawn thread");
}

pub fn timeout() -> bool {
    TIMEOUT.load(Ordering::SeqCst)
}
