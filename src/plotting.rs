use plotters::prelude::*;
use std::collections::HashMap;

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

    // Compute chance level loss: -log(1/97) ≈ 4.57
    let chance_loss = -(1.0 / 97.0_f64).ln();
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
                let color = if *freq == 97 {
                    RED.filled() // Highlight frequency 97 (the modulus)
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
        let chance_loss = -(1.0 / 97.0_f64).ln();
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

    // Compute chance level loss: -log(1/97) ≈ 4.57
    let chance_loss = -(1.0 / 97.0_f64).ln();
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
        let chance_loss = -(1.0 / 97.0_f64).ln();
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

/// Plot 7×7 embedding analysis grid
/// embeddings: [p, d] matrix where p=97 (number of tokens), d=embedding_dim
/// dimensions: indices of 7 embedding dimensions to visualize
pub fn plot_embedding_grid(
    embeddings: &[Vec<f64>],
    dimensions: &[usize; 7],
    output_path: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let p = embeddings.len(); // number of tokens (97)
    let point_size = 2; // small point size

    // Create large canvas for 7×7 grid
    let root = BitMapBackend::new(output_path, (2100, 2100)).into_drawing_area();
    root.fill(&BLACK)?; // Black background

    // Split into 7×7 grid
    let cell_areas = root.split_evenly((7, 7));

    for row in 0..7 {
        for col in 0..7 {
            let idx = row * 7 + col;
            let area = &cell_areas[idx];

            if col == 0 {
                // First column: embedding dimension vs token index
                let dim = dimensions[row];

                // Get dimension values for all tokens
                let values: Vec<(usize, f64)> = (0..p)
                    .map(|token| (token, embeddings[token][dim]))
                    .collect();

                // Find min/max for y-axis
                let min_val = values.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
                let max_val = values.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
                let margin = (max_val - min_val) * 0.1;

                let mut chart = ChartBuilder::on(area)
                    .margin(5)
                    .build_cartesian_2d(
                        0..p,
                        (min_val - margin)..(max_val + margin),
                    )?;

                // Subtle gray crosshairs at origin
                chart.configure_mesh()
                    .disable_x_mesh()
                    .disable_y_mesh()
                    .x_labels(0)
                    .y_labels(0)
                    .axis_style(RGBAColor(50, 50, 50, 0.3))
                    .draw()?;

                // Plot points colored by token value
                for (token, value) in values {
                    let color = token_color(token, p);
                    chart.draw_series(std::iter::once(Circle::new(
                        (token, value),
                        point_size,
                        color.filled(),
                    )))?;
                }
            } else {
                // Pairwise scatter plot
                let dim_x = dimensions[col];
                let dim_y = dimensions[row];

                // Extract x and y values
                let points: Vec<(f64, f64)> = (0..p)
                    .map(|token| (embeddings[token][dim_x], embeddings[token][dim_y]))
                    .collect();

                // Find min/max for axes
                let min_x = points.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
                let max_x = points.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
                let min_y = points.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
                let max_y = points.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);

                let margin_x = (max_x - min_x) * 0.1;
                let margin_y = (max_y - min_y) * 0.1;

                let mut chart = ChartBuilder::on(area)
                    .margin(5)
                    .build_cartesian_2d(
                        (min_x - margin_x)..(max_x + margin_x),
                        (min_y - margin_y)..(max_y + margin_y),
                    )?;

                // Subtle gray crosshairs at origin
                chart.configure_mesh()
                    .disable_x_mesh()
                    .disable_y_mesh()
                    .x_labels(0)
                    .y_labels(0)
                    .axis_style(RGBAColor(50, 50, 50, 0.3))
                    .draw()?;

                // Plot points colored by token value
                for token in 0..p {
                    let color = token_color(token, p);
                    let (x, y) = points[token];
                    chart.draw_series(std::iter::once(Circle::new(
                        (x, y),
                        point_size,
                        color.filled(),
                    )))?;
                }
            }
        }
    }

    root.present()?;
    println!("📊 Embedding grid plot saved to: {}", output_path);
    println!("   Title: {}", title);
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
