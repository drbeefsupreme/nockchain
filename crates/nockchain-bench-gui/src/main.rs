//! Nockchain Bench GUI
//!
//! A graphical interface for running Nockchain memory benchmarks.

use eframe::egui;
use nockchain_bench_gui::BenchApp;

fn main() -> eframe::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    tracing::info!("Starting Nockchain Bench GUI");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("Nockchain Bench"),
        ..Default::default()
    };

    eframe::run_native(
        "Nockchain Bench",
        options,
        Box::new(|_cc| {
            // Set up custom fonts and styling if needed
            let mut app = BenchApp::new();

            // Initialize the runner in a background thread
            app.init_runner();

            Ok(Box::new(app))
        }),
    )
}
