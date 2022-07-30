use image::{imageops, GrayImage, ImageBuffer, Luma};
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

pub fn window_crop(
    input_frame: &GrayImage,
    window_width: u32,
    window_height: u32,
    center: (u32, u32),
) -> GrayImage {
    let window = imageops::crop(
        &mut input_frame.clone(),
        center
            .0
            .saturating_sub(window_width / 2)
            .min(input_frame.width() - window_width),
        center
            .1
            .saturating_sub(window_height / 2)
            .min(input_frame.height() - window_height),
        window_width,
        window_height,
    )
    .to_image();

    return window;
}

pub fn to_imgbuf(buf: &Vec<f32>, width: u32, height: u32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    ImageBuffer::from_vec(width, height, buf.iter().map(|c| *c as u8).collect()).unwrap()
}

pub fn index_to_coords(width: u32, index: u32) -> (u32, u32) {
    // modulo/remainder ops are theoretically O(1)
    // checked_rem returns None if rhs == 0, which would indicate an upstream error (width == 0).
    let x = index.checked_rem(width).unwrap();

    // checked sub returns None if overflow occurred, which is also a panicable offense.
    // checked_div returns None if rhs == 0, which would indicate an upstream error (width == 0).
    let y = (index.checked_sub(x).unwrap()).checked_div(width).unwrap();
    return (x, y);
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

    #[test]
    fn window_crop_size() {
        let width: u32 = 4;
        let height: u32 = 8;
        let center = (0, 0);
        let image = GrayImage::new(32, 32);
        let cropped = window_crop(&image, width, height, center);

        assert_eq!(cropped.dimensions(), (width, height));
    }

    #[test]
    fn test_to_imgbuf() {
        let width: u32 = 4;
        let height: u32 = 8;
        let len: usize = (width * height) as usize;
        let zero_vec = vec![0.0; len];
        let img = to_imgbuf(&zero_vec, width, height);
        assert_eq!(img.dimensions(), (width, height));
    }

    #[test]
    fn test_index_to_coords() {
        let width: u32 = 4;
        let height: u32 = 8;
        assert_eq!(index_to_coords(width, 0), (0, 0));
        assert_eq!(index_to_coords(width, width), (0, 1));
        assert_eq!(
            index_to_coords(width, width * height - 1),
            (width - 1, height - 1)
        );
    }
}
