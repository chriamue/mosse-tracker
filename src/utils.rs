use image::{GrayImage};
use std::f32;

pub fn preprocess(image: &GrayImage) -> Vec<f32> {
    let mut prepped: Vec<f32> = image
        .pixels()
        // convert the pixel to u8 and then to f32
        .map(|p| p[0] as f32)
        // add 1, and take the natural logarithm
        .map(|p| (p + 1.0).ln())
        .collect();

    // normalize to mean = 0 (subtract image-wide mean from each pixel)
    let sum: f32 = prepped.iter().sum();
    let mean: f32 = sum / prepped.len() as f32;
    prepped.iter_mut().for_each(|p| *p = *p - mean);

    // normalize to norm = 1, if possible
    let u: f32 = prepped.iter().map(|a| a * a).sum();
    let norm = u.sqrt();
    if norm != 0.0 {
        prepped.iter_mut().for_each(|e| *e = *e / norm)
    }

    // multiply each pixel by a cosine window
    let (width, height) = image.dimensions();
    let mut position = 0;
    for i in 0..width {
        for j in 0..height {
            let cww = ((f32::consts::PI * i as f32) / (width - 1) as f32).sin();
            let cwh = ((f32::consts::PI * j as f32) / (height - 1) as f32).sin();
            prepped[position] = cww.min(cwh) * prepped[position];
            position += 1;
        }
    }

    return prepped;
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn preprocess_size() {
        let width: u32 = 4;
        let height: u32 = 8;
        let len: usize = (width * height) as usize;
        let image = GrayImage::new(width, height);
        let preprocessed = preprocess(&image);
        assert_eq!(preprocessed.len(), len);

        let zero_vec = vec![0.0; len];
        assert_eq!(preprocessed, zero_vec);
    }
}
