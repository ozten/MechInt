use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Represents a 3D surface dataset for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationSurface {
    /// Neuron index in the MLP layer
    pub neuron_index: usize,
    /// Modulus value (grid size will be modulus x modulus)
    pub modulus: usize,
    /// 2D grid of activation values: surface[x][y] = activation
    pub surface: Vec<Vec<f64>>,
}

impl ActivationSurface {
    pub fn new(neuron_index: usize, modulus: usize, surface: Vec<Vec<f64>>) -> Self {
        Self {
            neuron_index,
            modulus,
            surface,
        }
    }

    /// Save activation surface to JSON file
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize: {}", e))?;
        fs::write(path.as_ref(), json)
            .map_err(|e| format!("Failed to write file: {}", e))?;
        Ok(())
    }

    /// Load activation surface from JSON file
    /// NOTE: Reserved for loading activation surfaces from saved JSON files
    #[allow(dead_code)]
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let json = fs::read_to_string(path.as_ref())
            .map_err(|e| format!("Failed to read file: {}", e))?;
        let surface: ActivationSurface = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize: {}", e))?;
        Ok(surface)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_activation_surface_save_load() {
        let modulus = 5;
        let mut surface = vec![vec![0.0; modulus]; modulus];
        for x in 0..modulus {
            for y in 0..modulus {
                surface[x][y] = (x * modulus + y) as f64;
            }
        }

        let original = ActivationSurface::new(42, modulus, surface);

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        original.save_json(path).unwrap();
        let loaded = ActivationSurface::load_json(path).unwrap();

        assert_eq!(loaded.neuron_index, 42);
        assert_eq!(loaded.modulus, 5);
        assert_eq!(loaded.surface.len(), 5);
        assert_eq!(loaded.surface[0].len(), 5);
        assert_eq!(loaded.surface[2][3], 13.0);
    }

    #[test]
    fn test_activation_surface_json_format() {
        let surface = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let data = ActivationSurface::new(10, 2, surface);

        let json = serde_json::to_string_pretty(&data).unwrap();
        assert!(json.contains("\"neuron_index\": 10"));
        assert!(json.contains("\"modulus\": 2"));
        assert!(json.contains("\"surface\""));
    }
}
