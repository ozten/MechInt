use image::{ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut};
use plotters::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

use crate::data::ModularAdditionDataset;

/// Viridis colormap - maps value in [0, 1] to RGB
fn viridis_color(t: f64) -> RGBColor {
    // Viridis colormap approximation (purple → cyan → yellow)
    // Based on matplotlib's viridis
    let r = (0.267 + t * (0.329 - 0.267 + t * (0.984 - 0.329))) * 255.0;
    let g = (0.005 + t * (0.569 - 0.005 + t * (0.906 - 0.569))) * 255.0;
    let b = (0.329 + t * (0.758 - 0.329 - t * (0.758 - 0.121))) * 255.0;

    RGBColor(
        r.min(255.0).max(0.0) as u8,
        g.min(255.0).max(0.0) as u8,
        b.min(255.0).max(0.0) as u8,
    )
}

/// Map token value (0 to p-1) to viridis color
fn token_color(token_value: usize, p: usize) -> RGBColor {
    let t = token_value as f64 / (p - 1) as f64;
    viridis_color(t)
}

/// Plot training and validation loss over time
pub fn plot_loss_history_dual(
    train_loss: &[(usize, f64)], // (step, loss)
    val_loss: &[(usize, f64)],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_step = train_loss
        .iter()
        .chain(val_loss.iter())
        .map(|(s, _)| *s)
        .max()
        .unwrap_or(1);

    let max_loss = train_loss
        .iter()
        .chain(val_loss.iter())
        .map(|(_, l)| *l)
        .fold(0.0f64, f64::max)
        .max(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss Evolution: Train vs Validation", ("sans-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0usize..max_step, 0f64..max_loss)?;

    chart
        .configure_mesh()
        .x_desc("Training Step")
        .y_desc("Loss")
        .draw()?;

    // Plot training loss
    chart
        .draw_series(LineSeries::new(
            train_loss.iter().map(|(s, l)| (*s, *l)),
            &RED,
        ))?
        .label("Train Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot validation loss
    chart
        .draw_series(LineSeries::new(
            val_loss.iter().map(|(s, l)| (*s, *l)),
            &GREEN,
        ))?
        .label("Val Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    let modulus = ModularAdditionDataset::modulus() as f64;
    let chance_loss = -(1.0 / modulus).ln();
    chart
        .draw_series(LineSeries::new(
            vec![(0, chance_loss), (max_step, chance_loss)],
            &BLUE.mix(0.5),
        ))?
        .label("Chance Level")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE.mix(0.5)));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("📊 Loss plot saved to: {}", output_path);
    Ok(())
}

/// Plot training loss over time (legacy, single loss)
/// NOTE: Superseded by plot_loss_history_dual - kept for reference
#[allow(dead_code)]
pub fn plot_loss_history(
    snapshots: &[(usize, f64)], // (step, loss)
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_step = snapshots.iter().map(|(s, _)| *s).max().unwrap_or(1);
    let max_loss = snapshots
        .iter()
        .map(|(_, l)| *l)
        .fold(0.0f64, f64::max)
        .max(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss Evolution", ("sans-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0usize..max_step, 0f64..max_loss)?;

    chart
        .configure_mesh()
        .x_desc("Training Step")
        .y_desc("Loss")
        .draw()?;

    chart.draw_series(LineSeries::new(
        snapshots.iter().map(|(s, l)| (*s, *l)),
        &BLUE,
    ))?;

    root.present()?;
    println!("📊 Loss plot saved to: {}", output_path);
    Ok(())
}

/// Plot training and validation accuracy over time (combined)
pub fn plot_accuracy_history(
    train_acc: &[(usize, f32)], // (step, accuracy)
    val_acc: &[(usize, f32)],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_step = train_acc
        .iter()
        .chain(val_acc.iter())
        .map(|(s, _)| *s)
        .max()
        .unwrap_or(1);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Grokking Phenomenon: Train vs Validation Accuracy",
            ("sans-serif", 50).into_font(),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0usize..max_step, 0f32..105f32)?;

    chart
        .configure_mesh()
        .x_desc("Training Step")
        .y_desc("Accuracy (%)")
        .draw()?;

    // Plot training accuracy
    chart
        .draw_series(LineSeries::new(
            train_acc.iter().map(|(s, a)| (*s, a * 100.0)),
            &BLUE,
        ))?
        .label("Training Accuracy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Plot validation accuracy
    chart
        .draw_series(LineSeries::new(
            val_acc.iter().map(|(s, a)| (*s, a * 100.0)),
            &RED,
        ))?
        .label("Validation Accuracy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Add reference line at 90% (grokking threshold)
    chart.draw_series(LineSeries::new(
        vec![(0, 90.0), (max_step, 90.0)],
        &GREEN.mix(0.5),
    ))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("📊 Accuracy plot saved to: {}", output_path);
    Ok(())
}

/// Plot FFT dominant frequency distribution
pub fn plot_fft_frequency_distribution(
    dominant_frequencies: &[(usize, usize, f64)], // (token_id, frequency, magnitude)
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    // Count frequency occurrences
    let mut freq_counts: HashMap<usize, usize> = HashMap::new();
    for (_, freq, _) in dominant_frequencies {
        *freq_counts.entry(*freq).or_default() += 1;
    }

    let mut freq_vec: Vec<(usize, usize)> = freq_counts.into_iter().collect();
    freq_vec.sort_by_key(|(f, _)| *f);

    let max_freq = freq_vec.iter().map(|(f, _)| *f).max().unwrap_or(1);
    let max_count = freq_vec.iter().map(|(_, c)| *c).max().unwrap_or(1);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "FFT Dominant Frequency Distribution",
            ("sans-serif", 50).into_font(),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0usize..max_freq, 0usize..max_count + 1)?;

    chart
        .configure_mesh()
        .x_desc("Frequency")
        .y_desc("Number of Tokens")
        .draw()?;

    chart.draw_series(
        freq_vec
            .iter()
            .map(|(freq, count)| {
                let color = if *freq == ModularAdditionDataset::modulus() {
                    RED.filled() // Highlight frequency matching modulus
                } else {
                    BLUE.filled()
                };
                Rectangle::new([(*freq, 0), (*freq, *count)], color)
            }),
    )?;

    root.present()?;
    println!("📊 FFT frequency distribution plot saved to: {}", output_path);
    Ok(())
}

/// Plot FFT magnitude spectrum for selected tokens
pub fn plot_fft_spectra(
    token_spectra: &[(usize, Vec<f64>)], // (token_id, magnitude_spectrum)
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_freq = token_spectra
        .first()
        .map(|(_, spec)| spec.len())
        .unwrap_or(1);
    let max_magnitude = token_spectra
        .iter()
        .flat_map(|(_, spec)| spec.iter())
        .copied()
        .fold(0.0f64, f64::max)
        .max(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "FFT Magnitude Spectra (Selected Tokens)",
            ("sans-serif", 50).into_font(),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0usize..max_freq, 0f64..max_magnitude)?;

    chart
        .configure_mesh()
        .x_desc("Frequency")
        .y_desc("Magnitude")
        .draw()?;

    let colors = [&BLUE, &RED, &GREEN, &MAGENTA, &CYAN, &YELLOW];

    for (idx, (token_id, spectrum)) in token_spectra.iter().enumerate() {
        let color = colors[idx % colors.len()];
        chart
            .draw_series(LineSeries::new(
                spectrum.iter().enumerate().map(|(f, m)| (f, *m)),
                color,
            ))?
            .label(format!("Token {}", token_id))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("📊 FFT spectra plot saved to: {}", output_path);
    Ok(())
}

/// Plot combined grokking visualization (loss + accuracy on same timeline)
pub fn plot_grokking_combined(
    train_loss: &[(usize, f64)],
    val_loss: &[(usize, f64)],
    train_acc: &[(usize, f32)],
    val_acc: &[(usize, f32)],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1400, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(450);

    // Upper plot: Accuracy
    {
        let max_step = train_acc
            .iter()
            .chain(val_acc.iter())
            .map(|(s, _)| *s)
            .max()
            .unwrap_or(1);

        let mut chart = ChartBuilder::on(&upper)
            .caption(
                "Grokking: Accuracy Evolution",
                ("sans-serif", 40).into_font(),
            )
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0usize..max_step, 0f32..105f32)?;

        chart
            .configure_mesh()
            .x_desc("Training Step")
            .y_desc("Accuracy (%)")
            .draw()?;

        chart
            .draw_series(LineSeries::new(
                train_acc.iter().map(|(s, a)| (*s, a * 100.0)),
                &BLUE,
            ))?
            .label("Training")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart
            .draw_series(LineSeries::new(
                val_acc.iter().map(|(s, a)| (*s, a * 100.0)),
                &RED,
            ))?
            .label("Validation")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
    }

    // Lower plot: Loss (train and val)
    {
        let max_step = train_loss
            .iter()
            .chain(val_loss.iter())
            .map(|(s, _)| *s)
            .max()
            .unwrap_or(1);

        let max_loss = train_loss
            .iter()
            .chain(val_loss.iter())
            .map(|(_, l)| *l)
            .fold(0.0f64, f64::max)
            .max(1.0);

        let mut chart = ChartBuilder::on(&lower)
            .caption("Loss Evolution", ("sans-serif", 40).into_font())
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0usize..max_step, 0f64..max_loss)?;

        chart
            .configure_mesh()
            .x_desc("Training Step")
            .y_desc("Loss")
            .draw()?;

        // Train loss
        chart
            .draw_series(LineSeries::new(
                train_loss.iter().map(|(s, l)| (*s, *l)),
                &RED,
            ))?
            .label("Train")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        // Val loss
        chart
            .draw_series(LineSeries::new(
                val_loss.iter().map(|(s, l)| (*s, *l)),
                &GREEN,
            ))?
            .label("Val")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

        // Chance level
        let modulus = ModularAdditionDataset::modulus() as f64;
        let chance_loss = -(1.0 / modulus).ln();
        chart.draw_series(LineSeries::new(
            vec![(0, chance_loss), (max_step, chance_loss)],
            &BLUE.mix(0.5),
        ))?;

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
    }

    root.present()?;
    println!("📊 Combined grokking plot saved to: {}", output_path);
    Ok(())
}

/// Plot training and validation loss over time with LOG-SCALE x-axis
pub fn plot_loss_history_dual_logscale(
    train_loss: &[(usize, f64)], // (step, loss)
    val_loss: &[(usize, f64)],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_step = train_loss
        .iter()
        .chain(val_loss.iter())
        .map(|(s, _)| *s)
        .max()
        .unwrap_or(1);

    let max_loss = train_loss
        .iter()
        .chain(val_loss.iter())
        .map(|(_, l)| *l)
        .fold(0.0f64, f64::max)
        .max(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss Evolution (Log Scale): Train vs Validation", ("sans-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d((1.0..max_step as f64).log_scale(), 0f64..max_loss)?;

    chart
        .configure_mesh()
        .x_desc("Training Step (log scale)")
        .y_desc("Loss")
        .draw()?;

    // Plot training loss
    chart
        .draw_series(LineSeries::new(
            train_loss.iter().map(|(s, l)| (*s as f64, *l)),
            &RED,
        ))?
        .label("Train Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot validation loss
    chart
        .draw_series(LineSeries::new(
            val_loss.iter().map(|(s, l)| (*s as f64, *l)),
            &GREEN,
        ))?
        .label("Val Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    let modulus = ModularAdditionDataset::modulus() as f64;
    let chance_loss = -(1.0 / modulus).ln();
    chart
        .draw_series(LineSeries::new(
            vec![(1.0, chance_loss), (max_step as f64, chance_loss)],
            &BLUE.mix(0.5),
        ))?
        .label("Chance Level")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE.mix(0.5)));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("📊 Loss plot (log scale) saved to: {}", output_path);
    Ok(())
}

/// Plot training and validation accuracy over time with LOG-SCALE x-axis
pub fn plot_accuracy_history_logscale(
    train_acc: &[(usize, f32)], // (step, accuracy)
    val_acc: &[(usize, f32)],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_step = train_acc
        .iter()
        .chain(val_acc.iter())
        .map(|(s, _)| *s)
        .max()
        .unwrap_or(1);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Grokking Phenomenon (Log Scale): Train vs Validation Accuracy",
            ("sans-serif", 50).into_font(),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d((1.0..max_step as f64).log_scale(), 0f32..105f32)?;

    chart
        .configure_mesh()
        .x_desc("Training Step (log scale)")
        .y_desc("Accuracy (%)")
        .draw()?;

    // Plot training accuracy
    chart
        .draw_series(LineSeries::new(
            train_acc.iter().map(|(s, a)| (*s as f64, a * 100.0)),
            &BLUE,
        ))?
        .label("Training Accuracy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Plot validation accuracy
    chart
        .draw_series(LineSeries::new(
            val_acc.iter().map(|(s, a)| (*s as f64, a * 100.0)),
            &RED,
        ))?
        .label("Validation Accuracy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Add reference line at 90% (grokking threshold)
    chart.draw_series(LineSeries::new(
        vec![(1.0, 90.0), (max_step as f64, 90.0)],
        &GREEN.mix(0.5),
    ))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    println!("📊 Accuracy plot (log scale) saved to: {}", output_path);
    Ok(())
}

/// Plot combined grokking visualization with LOG-SCALE x-axis (loss + accuracy)
pub fn plot_grokking_combined_logscale(
    train_loss: &[(usize, f64)],
    val_loss: &[(usize, f64)],
    train_acc: &[(usize, f32)],
    val_acc: &[(usize, f32)],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1400, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(450);

    // Upper plot: Accuracy (log scale)
    {
        let max_step = train_acc
            .iter()
            .chain(val_acc.iter())
            .map(|(s, _)| *s)
            .max()
            .unwrap_or(1);

        let mut chart = ChartBuilder::on(&upper)
            .caption(
                "Grokking: Accuracy Evolution (Log Scale)",
                ("sans-serif", 40).into_font(),
            )
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d((1.0..max_step as f64).log_scale(), 0f32..105f32)?;

        chart
            .configure_mesh()
            .x_desc("Training Step (log scale)")
            .y_desc("Accuracy (%)")
            .draw()?;

        chart
            .draw_series(LineSeries::new(
                train_acc.iter().map(|(s, a)| (*s as f64, a * 100.0)),
                &BLUE,
            ))?
            .label("Training")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart
            .draw_series(LineSeries::new(
                val_acc.iter().map(|(s, a)| (*s as f64, a * 100.0)),
                &RED,
            ))?
            .label("Validation")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
    }

    // Lower plot: Loss (log scale)
    {
        let max_step = train_loss
            .iter()
            .chain(val_loss.iter())
            .map(|(s, _)| *s)
            .max()
            .unwrap_or(1);

        let max_loss = train_loss
            .iter()
            .chain(val_loss.iter())
            .map(|(_, l)| *l)
            .fold(0.0f64, f64::max)
            .max(1.0);

        let mut chart = ChartBuilder::on(&lower)
            .caption("Loss Evolution (Log Scale)", ("sans-serif", 40).into_font())
            .margin(15)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d((1.0..max_step as f64).log_scale(), 0f64..max_loss)?;

        chart
            .configure_mesh()
            .x_desc("Training Step (log scale)")
            .y_desc("Loss")
            .draw()?;

        // Train loss
        chart
            .draw_series(LineSeries::new(
                train_loss.iter().map(|(s, l)| (*s as f64, *l)),
                &RED,
            ))?
            .label("Train")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        // Val loss
        chart
            .draw_series(LineSeries::new(
                val_loss.iter().map(|(s, l)| (*s as f64, *l)),
                &GREEN,
            ))?
            .label("Val")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

        // Chance level
        let modulus = ModularAdditionDataset::modulus() as f64;
        let chance_loss = -(1.0 / modulus).ln();
        chart.draw_series(LineSeries::new(
            vec![(1.0, chance_loss), (max_step as f64, chance_loss)],
            &BLUE.mix(0.5),
        ))?;

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
    }

    root.present()?;
    println!("📊 Combined grokking plot (log scale) saved to: {}", output_path);
    Ok(())
}

/// Plot 3×3 embedding analysis grid (faster version)
/// embeddings: [p, d] matrix where p is the number of tokens, d=embedding_dim
/// dimensions: indices of 3 embedding dimensions to visualize
pub fn plot_embedding_grid_3x3(
    embeddings: &[Vec<f64>],
    dimensions: &[usize; 3],
    output_path: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let p = embeddings.len(); // number of tokens
    let cell_size: usize = 400;
    let grid_size: usize = 3;
    let total_size = cell_size * grid_size;

    let render_start = Instant::now();
    println!("   🎨 Creating {}×{} pixel canvas...", total_size, total_size);
    let mut img: RgbImage = ImageBuffer::from_pixel(
        total_size as u32,
        total_size as u32,
        Rgb([0, 0, 0]),
    );

    println!("   Rendering 3×3 grid ({} cells)...", 9);
    let mut cell_index = 0;
    for row in 0..grid_size {
        for col in 0..grid_size {
            let cell_start = Instant::now();
            cell_index += 1;
            print!("\r   Rendering 3×3 grid... {}/{}", cell_index, 9);
            use std::io::Write;
            std::io::stdout().flush().unwrap();

            let x_offset = col * cell_size;
            let y_offset = row * cell_size;

            if col == 0 {
                let dim = dimensions[row];
                render_dimension_plot(
                    &mut img,
                    embeddings,
                    dim,
                    p,
                    x_offset,
                    y_offset,
                    cell_size,
                );
            } else {
                let dim_x = dimensions[col];
                let dim_y = dimensions[row];
                render_scatter_plot(
                    &mut img,
                    embeddings,
                    dim_x,
                    dim_y,
                    p,
                    x_offset,
                    y_offset,
                    cell_size,
                );
            }

            let cell_elapsed = cell_start.elapsed();
            println!(
                "\r   Rendered cell {}/9 in {:.2?}           ",
                cell_index, cell_elapsed
            );
        }
    }

    println!();
    println!("   💾 Saving image to {}...", output_path);
    img.save(output_path)?;
    let render_elapsed = render_start.elapsed();
    println!("   ✅ Embedding grid rendered in {:.2?}", render_elapsed);
    println!("   📊 Embedding grid plot saved to: {}", output_path);
    println!("      Title: {}", title);
    Ok(())
}

/// Select 7 interesting dimensions from embeddings based on variance
pub fn select_interesting_dimensions(embeddings: &[Vec<f64>], n: usize) -> Vec<usize> {
    let d = embeddings[0].len(); // embedding dimension
    let p = embeddings.len(); // number of tokens

    // Compute variance for each dimension
    let mut variances: Vec<(usize, f64)> = Vec::with_capacity(d);

    for dim in 0..d {
        // Compute mean
        let mean: f64 = embeddings.iter().map(|emb| emb[dim]).sum::<f64>() / p as f64;

        // Compute variance
        let variance: f64 = embeddings
            .iter()
            .map(|emb| (emb[dim] - mean).powi(2))
            .sum::<f64>()
            / p as f64;

        variances.push((dim, variance));
    }

    // Sort by variance (descending)
    variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top n dimensions
    variances.iter().take(n).map(|(dim, _)| *dim).collect()
}

/// Convert viridis color to image::Rgb
// Helper functions for alternative visualizations - reserved for future use
#[allow(dead_code)]
fn viridis_to_rgb(t: f64) -> Rgb<u8> {
    let color = viridis_color(t);
    Rgb([color.0, color.1, color.2])
}

/// Map token value to RGB color
#[allow(dead_code)]
fn token_to_rgb(token_value: usize, p: usize) -> Rgb<u8> {
    let t = token_value as f64 / (p - 1) as f64;
    viridis_to_rgb(t)
}

/// Fast 7×7 embedding grid using direct pixel manipulation
/// NOTE: Alternative visualization approach - reserved for future use
pub fn plot_embedding_grid_fast(
    embeddings: &[Vec<f64>],
    dimensions: &[usize; 7],
    output_path: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let p = embeddings.len(); // number of tokens
    let cell_size: usize = 300; // pixels per cell
    let grid_size: usize = 7;
    let total_size = cell_size * grid_size;

    println!("   🎨 Creating {}×{} pixel canvas...", total_size, total_size);
    let mut img: RgbImage = ImageBuffer::from_pixel(total_size as u32, total_size as u32, Rgb([0, 0, 0]));

    println!("   🖌️  Rendering {} cells...", grid_size * grid_size);

    for row in 0..grid_size as usize {
        for col in 0..grid_size as usize {
            let cell_idx = row * grid_size as usize + col;
            if cell_idx % 10 == 0 {
                print!(".");
                use std::io::Write;
                std::io::stdout().flush().unwrap();
            }

            let x_offset = col * cell_size as usize;
            let y_offset = row * cell_size as usize;

            if col == 0 {
                // First column: dimension vs token index
                let dim = dimensions[row];
                render_dimension_plot(
                    &mut img,
                    embeddings,
                    dim,
                    p,
                    x_offset,
                    y_offset,
                    cell_size,
                );
            } else {
                // Pairwise scatter plot
                let dim_x = dimensions[col];
                let dim_y = dimensions[row];
                render_scatter_plot(
                    &mut img,
                    embeddings,
                    dim_x,
                    dim_y,
                    p,
                    x_offset,
                    y_offset,
                    cell_size,
                );
            }
        }
    }

    println!(); // newline after progress dots
    println!("   💾 Saving image to {}...", output_path);
    img.save(output_path)?;
    println!("   ✅ Embedding grid saved!");
    println!("      Title: {}", title);
    Ok(())
}

/// Render dimension vs index plot into a cell
fn render_dimension_plot(
    img: &mut RgbImage,
    embeddings: &[Vec<f64>],
    dim: usize,
    p: usize,
    x_offset: usize,
    y_offset: usize,
    cell_size: usize,
) {
    // Find min/max for this dimension
    let values: Vec<f64> = (0..p).map(|token| embeddings[token][dim]).collect();
    let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    if range == 0.0 {
        return; // Skip if no variation
    }

    let margin = 20;
    let plot_width = cell_size - 2 * margin;
    let plot_height = cell_size - 2 * margin;

    // Draw subtle crosshairs
    let mid_x = (x_offset + cell_size / 2) as f32;
    let mid_y = (y_offset + cell_size / 2) as f32;
    let gray = Rgb([30, 30, 30]);
    draw_line_segment_mut(
        img,
        ((x_offset + margin) as f32, mid_y),
        ((x_offset + cell_size - margin) as f32, mid_y),
        gray,
    );
    draw_line_segment_mut(
        img,
        (mid_x, (y_offset + margin) as f32),
        (mid_x, (y_offset + cell_size - margin) as f32),
        gray,
    );

    // Plot points
    for token in 0..p {
        let x = x_offset + margin + (token * plot_width / p);
        let normalized = (values[token] - min_val) / range;
        let y = y_offset + cell_size - margin - (normalized * plot_height as f64) as usize;

        let color = token_to_rgb(token, p);
        draw_filled_circle_mut(img, (x as i32, y as i32), 2, color);
    }
}

/// Render scatter plot into a cell
fn render_scatter_plot(
    img: &mut RgbImage,
    embeddings: &[Vec<f64>],
    dim_x: usize,
    dim_y: usize,
    p: usize,
    x_offset: usize,
    y_offset: usize,
    cell_size: usize,
) {
    // Find min/max for both dimensions
    let values_x: Vec<f64> = (0..p).map(|token| embeddings[token][dim_x]).collect();
    let values_y: Vec<f64> = (0..p).map(|token| embeddings[token][dim_y]).collect();

    let min_x = values_x.iter().copied().fold(f64::INFINITY, f64::min);
    let max_x = values_x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_y = values_y.iter().copied().fold(f64::INFINITY, f64::min);
    let max_y = values_y.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let range_x = max_x - min_x;
    let range_y = max_y - min_y;

    if range_x == 0.0 || range_y == 0.0 {
        return; // Skip if no variation
    }

    let margin = 20;
    let plot_width = cell_size - 2 * margin;
    let plot_height = cell_size - 2 * margin;

    // Draw subtle crosshairs at origin
    let mid_x = (x_offset + cell_size / 2) as f32;
    let mid_y = (y_offset + cell_size / 2) as f32;
    let gray = Rgb([30, 30, 30]);
    draw_line_segment_mut(
        img,
        ((x_offset + margin) as f32, mid_y),
        ((x_offset + cell_size - margin) as f32, mid_y),
        gray,
    );
    draw_line_segment_mut(
        img,
        (mid_x, (y_offset + margin) as f32),
        (mid_x, (y_offset + cell_size - margin) as f32),
        gray,
    );

    // Plot points
    for token in 0..p {
        let norm_x = (values_x[token] - min_x) / range_x;
        let norm_y = (values_y[token] - min_y) / range_y;

        let x = x_offset + margin + (norm_x * plot_width as f64) as usize;
        let y = y_offset + cell_size - margin - (norm_y * plot_height as f64) as usize;

        let color = token_to_rgb(token, p);
        draw_filled_circle_mut(img, (x as i32, y as i32), 2, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logscale_plotting_functions_exist() {
        // Verify that log-scale plotting functions exist and have the correct signature
        // This is a compile-time check that ensures the API is correct
        let _: fn(&[(usize, f64)], &[(usize, f64)], &str) -> Result<(), Box<dyn std::error::Error>> =
            plot_loss_history_dual_logscale;
        let _: fn(&[(usize, f32)], &[(usize, f32)], &str) -> Result<(), Box<dyn std::error::Error>> =
            plot_accuracy_history_logscale;
        let _: fn(&[(usize, f64)], &[(usize, f64)], &[(usize, f32)], &[(usize, f32)], &str) -> Result<(), Box<dyn std::error::Error>> =
            plot_grokking_combined_logscale;
    }

    #[test]
    fn snake_curve_can_be_verified_with_plotting_data() {
        use crate::verify::{verify_snake_curve_shape, SnakeCurveConfig};

        // Simulate grokking data that would produce a snake curve on log-scale plot
        let train_acc = vec![
            (1, 0.10),
            (10, 0.50),
            (100, 0.90),
            (500, 0.99),
            (1000, 0.995),
        ];
        let val_acc = vec![
            (1, 0.01),
            (10, 0.01),
            (100, 0.01),
            (500, 0.01),
            (1000, 0.01),
            (2000, 0.95),
        ];

        let config = SnakeCurveConfig::default_for_modulus(113);
        let report = verify_snake_curve_shape(&train_acc, &val_acc, &config)
            .expect("snake curve verification should pass on grokking data");

        // Verify the snake curve characteristics are present
        assert!(report.train_converged_by_step < 1000, "train should converge early");
        assert!(report.val_grok_step > 1000, "val should grok late");
        assert!(report.curve_shape_valid, "curve shape should be valid");
    }
}
