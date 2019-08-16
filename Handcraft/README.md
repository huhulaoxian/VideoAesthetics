# ImageAesthetics

Aesthetics in photography is how people usually characterize beauty in theis form of art.

High-level judgments, i.e., users' perception, of design have been shown to be correlated to low-level features of the appearance. Prior works suggest that the aesthetic and affective responses aroused by the visual appearnace of a design influence users' perception and experience. [1]

# 1. HandCraft feature on image 

# Available features
## Total 57 features on images

- Exposure of Light (f1)
- Saturation and Hue (f2_3)
- The Rule of Thirds (f4_6)
- Size and Aspect Ratio (f7_8)
- Low Depth of Field Indicators (f9_11)
- The effects of saturation and brightness on emotion (e.g., pleasure, arousal and dominance) (f12_14)
- Colorfulness (f15)
- W3C colors (f16_31)
- Tamura features (f32_34)
- Wavelet-based features (f35_46)
- Gray-Level Co-occurance Matrix (f47_f50)
- Dynamics (f51_f56)
- Level of Details (f57)

```python
python FeatureExtraction.py img_csv_path save_csv_path
```

### Color(23) <hr>
- Exposure of Light (f1)
- Saturation and Hue (f2_3)
- The effects of saturation and brightness on emotion (e.g., pleasure, arousal and dominance) (f12_14)
- Colorfulness (f15)
- W3C colors (f16_31)

### Texture(19) <hr>
- Tamura features (f32_34)
- Wavelet-based features (f35_46)
- Gray-Level Co-occurance Matrix (f47_f50)

### Composition(13) <hr>
- Level of Details (f57)
- Low Depth of Field Indicators (f9_11)
- The Rule of Thirds (f4_6)
- Dynamics (f51_f56)

### Others(2) <hr>
- Size and Aspect Ratio (f7_8)


## Color-based features
color has been found effective in evokeing emotions and is linked to users' perception and responses toward a product. In addition, Color has been shown to influence perceived trustworthiness, users' loyalty, and purchase intetion.[3]

## Texture-based features
Existing works have found the link between texture and visual perception. Various metrics of visual texture are examined for the effectiveness in infering perceived complexity, aesthetics, and interestingness of visual stiomuli. For example, richer texture reflects higher complexity and aesthetics is linearly correlated to visual texture.

## Organization-based features
How visual elements are organized not only affexts the efficiency of human mental process in perceiving visual information. but also links to users' preference toward a design. e.g., users perceive symmetrical design a highly appearling.



### 1. Wu features [1]

available features :  f1, f2_3, f12_14, f15, f16_31, f32_34, f35_46, f47_f50

#### Color-based features

- **The effects of saturation and brightness on emotion (e.g., pleasure, arousal and dominance)** (3) <br>
f12_14 : Pleasure = 0.69y + 0.22s, Arousal : -0.31y + 0.60s, Dominance : -0.76y + 0.32s


- **HSV statistics** <br>
f1_3 : HSV color values are alighned with human vision system and are widely used to quantify aesthetic and affective attributes.

- **Itten contrast** <br>
: Itten constrast formalizes color contrast in a way of how it induces emotion effect.

- **Color semantic histogram** <br>
:

- **Colorfulness[3]** (1) <br>
f15 : In their paper, Hasler and Süsstrunk first asked 20 non-expert participants to rate images on a 1-7 scale of colorfulness. This survey was conducted on a set of 84 images. The scale values were:


1. Not colorful
2. Slightly colorful
3. Moderately colorful
4. Averagely colorful
5. Quite colorful
6. Highly colorful
7. Extremely colorful

In order to set a baseline, the authors provided the participants with 4 example images and their corresponding colorfulness value from 1-7.

Through a series of experimental calculations, they derived a simple metric that correlated with the results of the viewers.

They found through these experiments that a simple opponent color space representation along with the mean and standard deviations of these values correlates to 95.3% of the survey data.

- **Dominant colors** (5, imperfect) <br>
Dominant colors are measure by extracting top N (=5) occurring colors using uniform color quantification.

- **W3C colors** (16) <br>
f16_31 : W3C colors measure the occurrence of the 16 basic nameable colors presented on a screen shot. This measure counts the percentage of pixcels close to one of the 16 colors that are semantically recognizable to users

#### Texture-based features

- **Tamura features** (3) <br>
f32_34 : Tamura texture features describe the coarseness, contrast and directionality of image, which are related to human psychological responses to visual perceptions.

- **Wavelet-based features** (12) <br>
f35_46 : One way to measure spatial smoothness in the image is to user Daubechies wavelet transform, which has often been used in the literature to characterize texture. Wavelet-based features allow a multi-scale partitioning across three color channels.

- **Gray-Level Co-occurance Matrix** (4) <br>
f47_f50 : GLCM analyzes texture information by calculating four statistical characteristics:contrast, correlation, energy, homogeneity

#### Organization-based features

- Symmetry
- Balance
- Equilibrium

```python
python WuFeatures img_path csv_path
```

### 2. Datta features [2]

available features : f1, f2_3, f4_6, f7_8, f9_11, f15, f35_46

- **Exposure of Light and Colorfulness(already mentioned above)** (1) <br>
f1 : (Exposure of Light) : Too much exposure (leadning to brighter shots) often yields lower quality pictures. Those that are too dark are often also not appealing. Thus light exposure can often be a good discriminant between high and low quality photographs.

- **Saturation and Hue** (2) <br>
f2_3 : (the average saturation and Hue) : Saturation indicates chromatic purity. Pure colors in a photo tend to be more appealing that dull or impurse ones. In natural out-door landscape photography, professionals use specialized film such as the Fuji Velvia to enhance the staturation to result in depper blue sky, greener grass, more vivid flowers.

- **The Rule of Thirds (3)** <br>
f4_6 : (Rule of Thirds): A very popular rule of thumnb in photography is the Rule of Thirds. The rule can be considered as a sloppy approximantion to the 'golden ratio' (about 0.618). It specifies that the main element, or the center of interest, in a photograph should lie at one of the four intersection. e.g., the eye of a man, were often placed alighned to one of the edges, on the inside. This implies that a large part of the main object often lies on the periphery or inside of the inner rectangle.

- **Familiarity Measure** <br>
(Familiarity Measure) :

- **Wavelet-based Texture** (12) <br>
f35_46 : (Wavelet-based feuatres) : One way to measure spatial smoothness in the image is to user Daubechies wavelet transform, which has often been used in the literature to characterize texture.

- **Size and Aspect Ratio** <br>
f7_8 : The size of an image has a good chance of affecting the photo ratings. It is well-known that 4:3 and 16:9 aspect ratios, which approximante the 'golden ratio' are chosen as standards for television screens.

- **Region Composition** <br>

- **Low Depth of Field Indicators** <br>
f9_11 : Professional photofraphers often reduce the depth of field (DOF) for shooting single objects by using larger aperture settings, macro lenses, or telephoto lenses. DOF is the range of distance from a camera that is acceptably sharp in the photograph. By reducinf the DOF, the photographer causes the background to blur, thus simplifying the image, reducing the busyness and drawing the attention of the observer to the object of interest, which is sharp.

- **Shpae Convexity** <br>

```python
python DattaFeatures img_path csv_path
```

### 3. MachajdikFeatures [4]

available features : f51_f56, f57

- **Dynamics** <br>
f51_f56 : Studies suggest that lines induce emotional effects. Horizontal lines are always associated with a static horizon and communicate calmmess, peacefulness and relaxation, vertical lines are clear and direct and communicate dignity and eternality, slant lines, on the othe hand, are unstable and communicate dynamism.

- **Level of Details** <br>
f57 : images with much detail generally produce a different psychological effect than minimalist compositions. 

## Reference

[1] Wu, Z., Kim, T., Li, Q., & Ma, X. (2019). Understanding and Modeling User-Perceived Brand Personality from Mobile Application UIs.

[2] Datta, R., Joshi, D., Li, J., & Wang, J. Z. (2006, May). Studying aesthetics in photographic images using a computational approach. In European conference on computer vision (pp. 288-301). Springer, Berlin, Heidelberg.

[3] Hasler, D., & Suesstrunk, S. E. (2003, June). Measuring colorfulness in natural images. In Human vision and electronic imaging VIII (Vol. 5007, pp. 87-96). International Society for Optics and Photonics.

[4] Machajdik, J., & Hanbury, A. (2010, October). Affective image classification using features inspired by psychology and art theory. In Proceedings of the 18th ACM international conference on Multimedia (pp. 83-92). ACM.

## Reference github

[1] https://github.com/Avikalp7/image-aesthetics-learning

[2] https://github.com/MarshalLeeeeee/Tamura-In-Python
