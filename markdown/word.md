<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>

# I. Introduction

## 1.1.1 Introduction to the Problem

The problem of depth estimation from images is a core challenge in the field of Computer Vision, with widespread applications across automated systems, self-driving cars, robotics, virtual reality (VR), augmented reality (AR), and more. By analyzing and determining the depth of objects within images, systems can better understand the spatial structure and depth of the real-world environment, allowing for intelligent decision-making and effective interactive responses.

Depth estimation from a single image (monocular depth estimation) is challenging because a single image provides only 2D data, lacking the stereo information available to human vision. Multi-image (stereo or multi-view depth estimation) approaches, on the other hand, leverage differences between viewpoints to reconstruct spatial depth. Both approaches demand complex and accurate algorithms to achieve optimal performance.

### 1.1.2 Practical Applications of Depth Estimation

- **Self-Driving Cars**: Depth estimation enables cars to detect the distance to objects, pedestrians, and other vehicles, assisting with collision avoidance and precise navigation.
- **Robotics**: Robots use depth information to navigate and manipulate objects accurately, essential for tasks across industrial, healthcare, and home environments.
- **AR/VR**: In VR and AR, depth perception enhances realism by positioning objects accurately in 3D space, allowing for more natural and immersive interactions.

## 1.2 Related Research and Achievements in the Past 5-7 Years

### 1.2.1 Research Trends

1. **Deep Learning-Based Approaches**

   a. **CNN-Based Monocular Depth Estimation**: Convolutional Neural Networks (CNNs) have significantly advanced single-image depth estimation. Architectures like U-Net, ResNet, and DenseNet help models learn crucial image features, producing detailed and accurate depth maps.

   b. **Transformer-Based Models**: In recent years, Transformers have been applied to depth estimation, with models like Vision Transformer (ViT) and Swin Transformer. These models leverage the spatial processing power of Transformers, improving depth estimation accuracy.

   c. **Hybrid Approaches**: Combining CNN and Transformer models has allowed for the best of both techniques: CNN’s spatial feature extraction and Transformers’ contextual processing capabilities.

2. **Unsupervised and Self-Supervised Learning**

   a. **Self-Supervised Depth Estimation**: A prominent research direction is self-supervised learning, where models learn from image pairs or videos without requiring ground-truth depth labels. This approach reduces the dependency on manually labeled data, making it adaptable to various environments and datasets.

   b. **Photometric Consistency Loss**: A popular technique in self-supervised learning is using photometric consistency loss, based on matching pixel intensities between images from different views. This improves model accuracy without requiring labeled depth data.

3. **Multi-Image Depth Estimation (Stereo/Multi-View)**

   a. **Stereo Matching Techniques**: For systems that capture multiple images, stereo matching techniques such as Semi-Global Matching (SGM), PatchMatch, and Cost Volume Processing allow models to extract depth information by comparing disparities between images.

   b. **Multi-View Stereo (MVS)**: This technique uses multiple perspectives to determine object depth more accurately than monocular methods. Popular methods include Patch-based Multi-View Stereo (PMVS) and COLMAP.

### 1.2.2 Key Results and Achievements

1. **Improvements in Accuracy and Performance**

   a. Newer models using CNNs and Transformers have achieved higher accuracy while reducing processing time and optimizing memory use, enabling depth estimation systems to perform efficiently on mobile and low-power devices. For example, advanced models like MiDaS and DPT (Depth Prediction Transformer) have successfully applied single-image depth estimation.

2. **Real-World Applications**

   a. Depth estimation algorithms have been successfully integrated into real-world applications, including Tesla's self-driving technology, Apple's Face ID, and Google's ARCore, allowing for realistic face recognition and depth simulation features for users.

3. **Challenges and Potential Solutions**

   a. Complex scenarios such as low light, transparent objects, or highly reflective surfaces still present challenges for depth estimation models. Current research focuses on developing advanced techniques like Image Denoising, Regularization, and Data Augmentation to enhance accuracy in complex environments.

# II. Mathematical background

Monocular vision can be deceiving.

![](image-7.png)

## 2.1. Epipolar Geometry
Epipolar geometry is the geometry of stereo vision. When two cameras view a 3D scene from two distinct positions, there are a number of geometric relations between the 3D points and their projections onto the 2D images that lead to constraints between the image points. These relations are derived based on the assumption that the cameras can be approximated by the pinhole camera model.

![](image-8.png)

### 2.1.1. Definition
The figure below depicts two pinhole cameras looking at point X. In real cameras, the image plane is actually behind the focal center. 
However, we simplified the problem by placing a *virtual* image plane in front of the camera, since real plane is symmetric about the focal center of the lens.

Each camera captures a 2D image of the 3D world. This conversion from 3D to 2D is referred to as a perspective projection and is described by the pinhole camera model.

![](image-11.png)


### 2.1.2. Epipole or epipolar point

Each center projects onto a distinct point into the other camera's image plane. These two image points, denoted by $e_L$ and $e_R$, are called epipoles or epipolar points.

Ep`ipolar point is the intersection of line connecting two camera's center $O_L$ and $O_R$ with their image plane.


### 2.1.3. Epipolar line
    
Line $O_L-X$ projected on right camera creating a line $e_R-X_R$ called the epipolar line.
Symmetrically, the line $O_R-X$ is seen by the right camera as a point and is seen as epipolar line $e_L-x_L$ by the left camera.

![](image.png)

The red lines are epipolar lines

When two image planes are parallel then the epipoles $e$ and $e'$ are located at infinity. 
Then the epipolar lines are parallel to $x$ axis of image.

![](image-2.png)

### 2.1.4. Epipolar plane
$X, O_L, O_R$ form a plane, called epipolar plane. The epipolar plane and all epipolar lines intersect the epipoles regardless of where $X$ is located.

### 2.1.5. Epipolar constraint

If the relative position of the two cameras is known, this leads to two important observations:

Assume the projection point $x_L$, the epipolar line $e_R-x_R$ and the point $X$ projects into the right image is known. A point $x_R$ which must lie on this particular epipolar line.

This provides an epipolar constraint: the projection of $X$ on the right camera plane xR must be contained in the $e_R-x_R$ epipolar line. All points $X$ e.g. $X_1, X_2, X_3$ on the $O_L–X_L$ line will verify that constraint. 

Epipolar constraints can also be described by the essential matrix or the fundamental matrix between the two cameras.

### 2.1.6. Disparity and Depth map

![](image-10.png)

The above diagram contains equivalent triangles. Writing their equivalent equations will yield us following result:

$$
\text{disparity}=x-x'=\frac{Bf}{Z}
$$
So the depth $Z$ would be:
$$
\text{depth}=\frac{x-x'}{Bf}
$$

Where:

- $x$ and $x'$ are the distance between image points and their corresponding camera center.

- $B$ is the baseline, distance between two camera center.

- $f$ is the focal of both camera (they should have the same).

So in short, the above equation says that the depth of a point in a scene is inversely proportional to the difference in distance of corresponding image points and their camera centers.

## 2.2. Essential matrix

### 2.2.1. Coordinate representation
This derivation follows the paper by Longuet-Higgins.

For simplicity, we assume all the cameras are **normalized** and project the 3D world onto their respective image planes. i.e $K = K' = I$.

Let the 3D coordinates of a point **P** be $(x_1,x_2,x_3)$ and $(x'_{1},x'_{2},x'_{3})$ relative to each cmera's coordinate system. 

The mapping from the coordinates of a 3D point P to the 2D image coordinates of the point's projection onto the image plane, according to the pinhole camera model, is given by:

$$
\left(\begin{array}{cc} 
y_1\\
y_2\\
1
\end{array}\right)
=
\frac{1}{x_3}
\left(\begin{array}{cc} 
x_1\\ 
x_2\\
x_3
\end{array}\right)
\quad \text{and} \quad
\left(\begin{array}{cc} 
y'_1\\
y'_2\\
1
\end{array}\right)
=
\frac{1}{x'_3}
\left(\begin{array}{cc} 
x'_1\\ 
x'_2\\
x'_3
\end{array}\right)
$$ 

More compactly as:

$$
\begin{align}
y = \frac{1}{x_3} {x} \quad \text{and} \quad y' = \frac{1}{x_3}{x} 
\end{align}
$$
 
Where $y$ are the image 2D coordinate, $x$ are real proper 3D cooordinates but in two different coordinate systems.

### 2.2.2. Set up camera framework

![](image-3.png)

Let further assume world reference coordinate is associated with the first camera with the second camera offset by a $3\times3$ rotaion matrix $R$  and 3 dimensional translation matrix $t$. This implies:

$$
\begin{align}
x'=Rx+t
\end{align}
$$

And the camera matrix will be:

$$
\begin{align}
M = P[I \quad 0] \quad \text{and} \quad M' = P'[R^T \quad R^T-t]
\end{align}
$$

Or (because of cameras are normalized): 

$$
\begin{align}
M = [I \quad 0] \quad \text{and} \quad M' = [R^T \quad R^T-t]' 
\end{align}
$$

### 2.2.3. Essential matrix derivation

Since the vectors $Rx' + t$ and $t$ lie in the epipolar plane, then if we take the cross product of $t \times (Rx' + t) = t \times (Rx')$, we will get a vector normal to the epipolar plane. This also means that $x$ which lies in the epipolar plane is normal to $t \times (Rx')$, giving us the constraint that their dot product is zero:

$$
\begin {align}
{x}.[t \times (R{x}')]=0
\end {align}
$$


**Reminder from linear algebra:**
The cross product between any two vectors a and b as a matrix-vector multiplication:
$$
\begin {align}
a \times b =
\left(\begin{array}{cc} 
0 & -a_z & a_y\\ 
a_z & 0 & -a_x \\
-a_y & a_x & 0
\end{array}\right)
\left(\begin{array}{cc} 
b_x\\ 
b_y\\
b_z
\end{array}\right) =
[a_\times]b
\end {align}
$$

Combining this expression with Equation 6, we can convert the cross
product term into matrix multiplication, giving:

$$
x^T[t_\times](Rx')=0  
$$

$$
\begin {align}
x^T[t_\times]Rx'=0
\end {align}
$$

Then, the **Essential Matrix** is $E = [t\times]R$ creating a com-
pact expression for the epipolar constraint:
$$
\begin {align}
x^TEx'=0
\end {align}
$$

The Essential matrix is a 3 × 3 matrix that contains 5 degrees of freedom. It has rank 2 and is singular.

### 2.2.4. Essential matrix mapping

Different to a homography which maps a point to a point, an essential matrix maps a **point** to a **line**.
Furthermore, let's consider an epipolar line $l$, with the form of $ax+by+c=0$, or in vector form:

$$
\begin{aligned}
l=
\left(\begin{array}{c} 
a\\
b\\
c
\end{array}\right) \quad \text{and} \quad x^T l=0
\end{aligned}
$$

Then it is easy to see that, from Equation 8:

$$
\begin{align}
l=Ex' \quad \text{and} \quad l'=E^Tx
\end{align}
$$


### 2.2.5. Essential matrix kernel
Since every lines on image plane pass epipolar. So $e^Tl=0$, combine with Equation 9, we have $e^Tl=0=e^TEx'$. Furthermore, $e^TE=(E^Te)^T=l'^T$ thus is normal with $x'$. 

In short, essential kernel defines the epipole:
$$
\begin {align}
e^TE=0 \quad and \quad Ee'=0
\end {align}
$$
(points in normalized camera coordinates)

## 2.3. Fundamental matrix
### 2.3.1. Camera matrix
How do you generalize to uncalibrated cameras? Recall the Equation 3:
$$
\begin{align}
M = K[I & 0] & and & M' = K'[R^T & R^T-t]' 
\end{align}
$$
First, we must definde $X_W$ is a point from 3d world. We get two projected point on cameras:

$$
\begin {aligned}
X_0=P X_W\\
X_0'=P'X_W
\end {aligned}
$$

Say we have canonical cameras transform space by a general homography matrix $H = I$, 
then we have projections of $X_W$ to the corresponding camera images.

$$
\begin {aligned}
x = K^-1 X_W\\
x'=K^-1'X_W'
\end {aligned}
$$

### 2.3.2. Fundamental matrix derivation
Recall that in the canonical case from Equation 7:
$$
\begin {align}
x^T[t_\times]Rx'=0
\end {align}
$$

By substituting in the values of $x$ and $x'$, we get:
$$
\begin {align}
x^T K^{-T}  [t_\times] R K'^{-1} x' =0
\end {align}
$$

Let the matrix $F = K^{-T}  [t_\times] R K'^{-1}$ as the **Fundamental Matrix** which acts the same to the Essential matrix from previous but also encondes information about the camera matrices $K$ and $K'$ and the relative translation T and rotation R between the cameras.

Therefore, it is also useful in computing the epipolar lines associated with p and p′, even when the camera matrices K, K′ and the transformation R, T are unknown.

### 2.3.3. Propertises of fundamental matrix

Similar to the Essential matrix, we can compute the epipolar lines $l'=F^T x$ and $l=Fx'$ from just the fundamental matrix and the corresponding points.

Fundamental matrix contains 7 degrees of freedom, while Essential matrix’s 5 degrees of freedom.

If we know the fundamental matrix, then simply knowing a point in an image
gives us an easy constraint (the epipolar line) of the corresponding point in the other image. Therefore, without knowing the actual position of $X_W$ in 3D space, or any of the extrinsic or intrinsic characteristics of the cameras, we can establish a relationship between any $x$ and $x'$.

## 2.4. The Eight-Point algorithm
### 2.4.1. Formulating a homogeneous linear equation

With each correspondent $x$ and $x'$

$$\begin{aligned}
x=
\left(\begin{array}{cc} 
x_1\\ 
x_2\\
1
\end{array}\right)
\quad \text{and} \quad
x'=
\left(\begin{array}{cc} 
x'_1\\ 
x'_2\\
1
\end{array}\right)
\quad \text{and} \quad
F = 
\left(\begin{array}{cc} 
f_{11} & f_{12} & f_{13}\\
f_{21} & f_{22} & f_{23}\\
f_{31} & f_{32} & f_{33}
\end{array}\right)
\end{aligned}$$


The constraint can be rewritten as:

$$\left(\begin{array}{cc} 
x'_1x_1 & x'_1x_2 & x'_1 &
x'_2x_1 & x'_2x_2 & x'_2 &
x_1 & x_2 & 1
\end{array}\right)
\begin {aligned}
\left(\begin{array}{cc} 
f_{11} \\ f_{12} \\ f_{13}\\
f_{21} \\ f_{22} \\ f_{23}\\
f_{31} \\ f_{32} \\ f_{33}
\end{array}\right)
=0
\end {aligned}$$

That is $f$ represents the flatten **Fundamental matrix** vector and 
this vector must be othorgonal to vector $\mathbf{\bar{x}} = x'x^T$.

Each pair of corresponding image points produces a vector $\mathbf{\bar{x}}$. 
Given a set of 3D points $\mathbf{X_W}$ corresponding to a set of vector $\mathbf{\bar{x}}$ 
and all of them must satisfy:

$$
\mathbf{\bar{x}} \cdot f = 0
$$


Collect $N$ vector $\mathbf{\bar{x}}$ as the row of matrix $\mathbf{X}$ and:
$$
Xf=0
$$

Where $\mathbf{X}$ is a $N \times 9$ matrix with $N \ge 8$.

### 2.4.2. Solving the equation
In pracitce, there are noise so solution vector f is defined only up to an unknown scale.
So it is better to use more than eight correspondences and create a larger $X$. 
Furthermore, $X$ is often rank-deficient, so we approximate $f$ by **Linear least squares**:

$$
\begin{align}
\begin {split}
    \min_f & \quad \lVert Xf\rVert \\
    \text{subject to} & \quad \lVert f\rVert =1
\end{split}
\end{align} 
$$

The subject is to avoid the trivial solution f.

The solution to this optimize problem can be found by Singular Value Decomposition (SVD).
$f$ is the right singular vector corresponding to the smallest singular value of $X$. A reshape
of this $f$ into $3 \times 3$ matrix give result called as $\mathbf{F_{est}}$.

### 2.4.3. Enforcing the internal constraint
An important property of the fundamental matrix is that it is singular, in fact of rank 2. 
Furthermore, the left and right null spaces ($e$ and $e'$) of $F$ are generated by the vectors representing the two epipoles in the images i.e $dim Null(F) = 1$. 
However, often, dealing with noisy image gives the result $\mathbf{F_{est}}$ from Equation 14 usually does not have rank 2.

We find a best rank-2 matrix approximation of F by the mean of:

$$
\begin{align}
    \begin{split}
    \min_F & \quad \lVert F_{est} - F\rVert \\
    \text{subject to} & \quad \det F =0
    \end {split}
\end{align} 
$$

The constrain is to make $F$ is singular.

This problem is solved again by SVD, where $F = U\Sigma V^T$ then the best rank-2 approximation is found by:

$$
F = U
\begin{bmatrix}
\Sigma_1 & 0 & 0 \\
0 & \Sigma_2 & 0 \\
0 & 0 & 0 
\end{bmatrix}
V^T 
$$


### 2.4.4. Normalized algorithm
#### 2.4.4.1. Problems

The problem of the standard algorithm is that $X$ is often ill-conditioned for SVD. For SVD to work properly, $X$ should have differences between singular values not too large.

However, correspondences coordinate $(x,y,1)$ will often have
extremely large values in the first and second compared to the third $(\bar{x}=(1920,1080,1))$ due to large pixel range of mordern digital camera.

Furthermore, if the image points used to construct $X$ lie in a relatively small region of the image $((700,700)\pm100)$, then $x$ and $x'$ are relatively similar, resulting in $X$ has one ery large singular value, with the rest relatively small.

#### 2.4.4.2. Solution
To solve this, map each coordinate system of two images independently into a new system satisfying two conditions:

- The origin of the new system should be at the centroid (center of gravity) of the image points. This is accomplished by translating original origin to new one.

- After the translation, the coordinates have to be uniformed so that the mean of distance from each points to the origin equals $\sqrt 2$. This can e done by the scaling factor for each respective image

$$
\displaystyle \sqrt \frac{2N}{\displaystyle \sum_{i=1}^N \lVert x_i-\mu \rVert^2}
$$

Afterwards, a distinct coordinate transformation for each of the two images. We obtain a new homogeneous image:

$$
\begin {align*}
\bar x = Tx \\
\bar x'= T'x'
\end{align*}
$$

This normalization is only dependent on the image points which are used in a single image and is, in general, distinct from normalized image coordinates produced by a normalized camera.

Note that we overload the notations because of the obivious relations.

The epipolar constraint based on the fundamental matrix can now be rewritten as:

$$
x^TFx' =\bar x'^T T'^{-T} F T^{-1} T \bar x = \bar x'^T \bar F \bar x = 0
$$

Where $\bar F = T'^{-T} F T^{-1} T$. 

This means that it is possible to use the normalized homogeneous image coordinates, $\bar x$ and $\bar x'$, 
to estimate the transformed fundamental matrix $\bar F$ using the basic eight-point algorithm described above.

The solution $\bar F$ is now more well-defined from the homogeneous equation $\bar X \bar F$ than $F$ is relative to $X$.
Once $\bar F$ has been determined, we can de-normalized to give $F$ by:

$$
F = T'^T\bar FT
$$

# IIII. Solution

Most of the algorithm used in our codes are described above.
First, we downsize the image (optional, to improve running time). Then using Oriented FAST and Rotated BRIEF (ORB) descriptor from *cv2.ORB_create* to find and matched feature points between two images.
From the matched feature points, we use RANSAC to randomize the inputs to calculate Fundamental matrix by [8-points algorithm](#4-the-eight-point-algorithm) to reduce noises.

We calculate disparity by [Sum of Squared Differences](#321-def-sum_of_squared_diffpixel_vals_1-pixel_vals_2), then using the baseline and focal length from data calibration file to calculate the depth image.

![](flow-chart.drawio.png)

For better running time, we also use numba to optimize Python functions with decorator *@jit(nopython=True)*, which speed up running time from 5 minutes to 1 minutes compared to pure python. 

## 3.1. calibration.py

In this file, we compute the feature points, then estimate the fundamental and essential matrix from them. Furthermore, we find the correct camera pose (Rotation and Translation) from the $E$ matrix.

### 3.1.1. def draw_keypoints_and_match(img1: MatLike, img2: MatLike, nfeatures: int = 500) -> Tuple[NDArray, NDArray, MatLike]: ...

Finding keypoints using ORB feature detection and descriptors in the image and find best matches using brute force based matcher.

### 3.1.2. def compute_Fundamental_matrix(kp1_list: NDArray, kp2_list: NDArray) -> NDArray: ...

Calculate the $F$ matrix from a set of 8 points using SVD.
Furthermore, the rank of $F$ matrix is reduced from 3 to 2 to make the epilines converge.

More details at section [8 points algorithm](#42-solving-the-equation)

### 3.1.3. def RANSAC_F_mat(kp1_list: NDArray, kp2_list: NDArray, max_inliers=20, threshold=0.05, max_iter=1000) -> NDArray: ...

Shortlist the best F matrix using RANSAC based on the number of inliers.

RANSAC randomize 8 feature points to compute $F$ and calculate errors in estimating the points. Then we classify inliers with predefined threshold. Best $F$ is one with the most inliers.

### 3.1.4. def compute_Essential_matrix(F_mat: NDArray, K1: NDArray, K2: NDArray) -> NDArray: ...

Calculation of the Essential matrix with $F$, $K_1$ and $K_2$ camera intrinsic.

More details at section [Essential-matrix](#2-essential-matrix)

### 3.1.5. def drawlines(img1src: MatLike, img2src: MatLike, lines, pts1src, pts2src, random_seed=0) -> tuple[MatLike, MatLike]: ...

Visualize the epilines on both the images

## 3.2. correspondence.py

### 3.2.1. def sum_of_squared_diff(pixel_vals_1, pixel_vals_2):

We use Sum of Squared Differences (SSD) formular:

$$\text{SSD} = \displaystyle{\sum_{i,j} {(I_1(i,j)- I_2(i,j))^2}}
$$

### 3.2.2. def block_comparison(y, x, block_left, right_array, block_size, x_search_block_size, y_search_block_size):

Block comparison function to find minimum SSD match.
Two image block are correspodence if their SSD is the minimum.

### 3.2.3. def ssd_correspondence(img1, img2):

Correspondence applied on the whole image to compute the disparity map and finally disparity map is scaled by min max scaling.

## 3.3. depth.py

### 3.3.1. def disparity_to_depth(baseline, f, img):

$$\text{depth} = \frac{Bf}{x-x'}$$

More details at section [Depth Map](#16-disparity-and-depth-map) 


## 4. Notebooks:
Mainly use for testing/ reviewing and produces image results.

# IIII. Evaluation

## 4.1. The results achieved

### 4.1.1 Dataset used (Middle Blurry Q-Size)

We here use the middle blurry, quarter-size datasets to accommodate Python's performance. From the looks of features matching, we can see that the cameras are calibrated and two image planes has been already parallel, so no rectification is needed:

![](image-24.png)

Our main error measurement for two given matrix is RMSE:

$$
\text{RMSE} = \sqrt{\frac{1}{mn} \sum_{i=1}^{m} \sum_{j=1}^{n} (I_{i,j} - \hat{I}_{i,j})^2}
$$

Where:

- $m$ and $n$ are the dimensions of the image.

- $I_{i,j}$ is the actual pixel value at position $i,j$

- $\hat{I}_{i,j}$ is the predicted pixel value at position $i,j$

### 4.1.2. Esimating Fundamental matrix

Our estimation of Fundamental matrix is very close to *cv2.findFundamentalMat* for the first 5 datasets: **mean** = 0.053111; **min** = 0.00034; **max** = 0.120961. 
For the last dataset, the rmse error is 7e+21:

![](image-25.png)

We suspect the logo on the bag creates too many features outliers that are too close to each other, thus yield very distorted results:

```python 
>>> cv2.findFundamentalMat(list_kp1, list_kp2)[0]
array([[-5.12793973e-08, -1.42405116e-04,  3.62484045e-02],
       [ 1.42119735e-04, -5.70761988e-06,  1.85833981e+11],
       [-3.59749826e-02, -1.85833981e+11,  1.00000000e+00]]) 

>>> RANSAC_F_mat(list_kp1, list_kp2)
array([[ 1.42808893e-17,  2.60663832e-09, -1.00222640e-06],
       [-2.49459138e-09,  9.15801324e-11,  7.07106677e-01],
       [ 9.55271472e-07, -7.07106882e-01,  6.59325435e-05]])
```

Since our algorithm is based on randomizing matched feature points, the noises produces from that did not affect the results.

### 4.1.3. Disparity map

For images with not many sudden depth and the majority is smooth surface, our RMSE error is quite low:

![](image-26.png)

While many uneven, discontinuous surface like on a bike when its engine has many smaller compartments inside:

![](image-27.png)


Overal, our RMSE range from 14221 to 3445. The average score in 6 datasets is 7532.

There are a lot of noises in depth-discontinuity area, and a border 15 padding because of SSD block-matching algorithm. The noises can be rectified by using Fast Global Smoother Filter for post-filtering, the reasons for this problem will be discussed later.

> **Note:**  All of the images, results and further details can be found at *evaludation.ipynb*.

> **Note:**  The number on the image label is RMSE error compared to ground truth

> **Note**  Any $\text{inf}$ values in ground truth has to be set to *ndisp* to avoid $\text{inf}$ results.

## 4.2. Compare with the results of the group on the same topic

I have no frien :<

## 4.3. Previous studies

### 4.3.1. Calculating correspondences (SGM)

We used Sum of Squared Differences (SSD) to calculate the coressponding block.
Howvever, this function inherently assume there is no sudden depth variations and occlusions. Furthermore, this assume continuous or functionally variable disparities within the correlation window. Thus often produce very noisy results, jumping and gap pixels.

![](image-15.png)

One portion of the scene
may obscure other portions in either or both of the two cameras. Ex: $B-C-s_1$.

The Semi-Global Matching (SGM) implements Mutual Information cost function. The pixelwise cost and the smoothness constraints are expressed by defining the energy E(D) that depends on the disparity 
image D:

$$\displaystyle {E(D) = \sum_{p} \left( C(p, D_{p}) + \sum_{q \in N_{p}} P_{1} T[|D_{p} - D_{q}| = 1] + \sum_{q \in N_{p}} P_{2} T[|D_{p} - D_{q}| > 1] \right)}$$

Where:

- $D$ represents the disparity map.

- $C(p, D_p)$ is the data term that measures the fidelity of the disparity map $D$ at pixel $p$.

- $N_p$ denotes the neighborhood of pixel $p$.

- $T$ is an indicator function.

- First term is the similarity cost, the sum of all pixel matching costs for the disparities of $D$.

- $P_1$ penalty for small disparity changes  (1 pixel).

- $P_2$ penalty is constant and larger for all larger disparity changes.

Using a lower penalty for small changes permits an adaptation to slanted or curved surfaces. The constant penalty for all larger changes (i.e. independent of their size) preserves discontinuities. While SSD is too simple to capture these characters.

Generally, SGM outperforms us, here is the comparision:

![](image-17.png)

It is clearly at our algorithm is sensitive to noise. At data 2, our algorithm outperforms SGM when the surface are generally smooth and there are no sudden depth.

![](image-18.png)

While at data 3, there is a lot of pipe obstructing each others, meaning there is a lot of sudden depth.

![](image-19.png)


This module is available at *cv2.StereoBM_create*

### 4.3.2. Disaprity post Filtering (Fast Global Smoother Filter)

Its objective is to enhance the quality of the input low-resolution depth map by increasing its spatial resolution. Low-resolution depth map and its associated high-resolution color image are used as inputs. It has the ability to preserve edges, which is important because one object should has a continuous disparity.


Fast Global Smoother Filter first take the left image as **guidance image** $g$, an **input image** $f$ and produces a **desired output** $u$. It tries to minimize the following Weighted Least Square (WSL) equation:

$$
E(u) = \sum_{p} (u_p - f_p)^2 + \lambda \sum_{q \in N(p)} w_{p,q}(g) (u_p - u_q)^2
$$

Where:

- $\lambda$ controls the balance between the two terms. Increasing $\lambda$ results in more smoothing in the output u. 

- The first term expresses the similarity between outout and input.

- $N(p)$ are neighbour pixels if pixel $p$.

- $w_{p,q}(g)$ expresses the "inverse" similarity between pixel $p$ and $q$. For example: $w_{p,q}(g) = \exp\left( - \frac{\| g_p - g_q \|}{\sigma_c} \right)$

- The second term expresses the similarity between similarity between pixel $p$ and its neighbor $N(p)$ in output $u$.

When neighbour in $g$ is large (possibly edge), $w_{p,q}(g)$ is small and thus, neighbour in $u$ can be large without affecting the overal energy. While neighbour in $g$ is small, forcing neighbour in $u$ to be also small. Overall, this try to preserve the edges while smoothing low-variance, smooth surface.

Here, we set parameter *sigma_color = 1.5* while use grid search strategy to find the best *lambda_value = 8273000*. The noise reduction is obivious:

![](image-22.png)

Comparing with Gaussian filter and Mean filter, we got:

![](image-20.png)

The most impactful scenarios is Data 3, where there are a lot of sudden depth:

![](image-21.png)

~~the proof is left to the reader~~. 
- Gaussian blurs edges, which we do not want because one object should has continuous and similar disparity, which helps them stand out. 

- Median filter is not inherently continuous while the noises are usually big enough to be spilled further.

This module is available at *cv2.ximgproc.createFastGlobalSmootherFilter*

### 4.3.3. Cencus Transform

Cencus transform represents a block as the relationship between the center and its neighbour for matching instead of the original. Then hamming distance is used to calculate the similarity between blocks.
Our implementation of cencus trasnform each neighbour pixel $q$ of $p$ (in gray scale) to:

- $1$ if $q>p$

- $0$ if $q=p$

- $-1$ if $q<p$

For example:

$$
\begin{bmatrix}
  34 & 123 & 87 \\
  210 & 56 & 78 \\
  99 & 45 & 200
\end{bmatrix}
\rightarrow
\begin{bmatrix}
  -1 & 1 & 1 \\
  1 & 0 & 1 \\
  1 & -1 & 1
\end{bmatrix}
\quad \text{and} \quad
\begin{bmatrix}
  12 & 234 & 56 \\
  78 & 90 & 123 \\
  45 & 67 & 189
\end{bmatrix}
\rightarrow
\begin{bmatrix}
  -1 & 1 & -1 \\
  -1 & 0 & 1 \\
  -1 & -1 & 1
\end{bmatrix}
$$

Their Hamming distance is 3.

Here are some comparision of results between our algorithm and cencus. We used block size of 5, any larger block size would yeild much more noises:

![](image-29.png)

![](image-30.png)

Unfiltered cencus performs the worst because of noises, thus when being smoothed by Fast Global Smoothing, its accuracy is improved drastically, but not as much as our algorithm.

![](image-28.png)


# References

https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
https://cmsc426.github.io/sfm/#essential
https://en.wikipedia.org/wiki/Epipolar_geometry
https://en.wikipedia.org/wiki/Essential_matrix
https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)

For 4. Evaluation

https://core.ac.uk/download/pdf/11134866.pdf
https://engineering.purdue.edu/kak/Tutorials/SemiGlobalMatching.pdf

WSL Filter:
https://publish.illinois.edu/visual-modeling-and-analytics/files/2014/10/FGS-TIP.pdf
