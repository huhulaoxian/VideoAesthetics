# Video key frame extraction


## 1. Key frame extraction using difference of RGB color histrogram

```python
python RGBdetection.py (video path) (file path for extracted frame) (Paremeter to frame you want from video)
```

## 2. Key frame extraction using difference of RGB values

```python
python RGBdetection_colorhist.py (video path) (file path for extracted frame) (Paremeter to frame you want from video)
```

## 3. Key frame extraction using difference of RGB color histogram with Threshold

Sheena, C. V., & Narayanan, N. K. (2015). Key-frame extraction by analysis of histograms of video frames using statistical methods. Procedia Computer Science, 70, 36-40.

```python
python RGBdetection_colorhist.py (video path) (file path for extracted frame) (option - Paremeter to frame you want from video -- default : none)
```

## 4. VSUMM

De Avila, S. E. F., Lopes, A. P. B., da Luz Jr, A., & de Albuquerque Araújo, A. (2011). VSUMM: A mechanism designed to produce static video summaries and a novel evaluation method. Pattern Recognition Letters, 32(1), 56-68.

```python
python vsumm.py (video path) (file path for extracted frame) (option - Paremeter to frame you want from video -- default : none) - Under construction
```

## Additional information
1. Difference in LUV space

- The choice of LUV colorspace is mainly due to how it differentiates illuminance from chromaticity which will not be available in RGB colorspace.
Another main reason better explained under this link is as follows:

- More specifically 'Luv' was designed to be 'perceptually linear'. That is that a small change in color in one part of the colorspace looks to be about the same, as a similar change in another part of the colorspace. This makes LUV colorspace much better suited for image difference comparisons.
: https://photo.stackexchange.com/questions/67933/when-to-use-luv-and-not-rgb-colourspaces


## Reference
KeyFrame extraction using histogram : https://github.com/amanwalia92/KeyFramesExtraction

VSUMM : https://github.com/susilvaalmeida/vsumm
