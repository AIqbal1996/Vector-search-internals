# Mathematical Foundations: Vector Search Internals

## 1. Geometric Space, Coordinates, and Vectors

In linear algebra, an $N$-dimensional vector $v \in \mathbb{R}^D$ represents a directed quantity defined by an ordered tuple of real numbers:
$$v = \begin{bmatrix} v_1 & v_2 & \dots & v_D \end{bmatrix}^T$$

In modern AI systems, high-dimensional vectors (known as **embeddings**) are produced by deep neural networks (e.g., Google's Gemini Embedding 2 with $D = 3072$ dimensions). These vectors project complex semantic concepts into a continuous geometric vector space, where concepts with similar meanings are positioned closely together.

---

## 2. Mathematical Definitions & Scratch Formulations

### 2.1 Vector Magnitude (Euclidean L2 Norm)
The magnitude or length of a vector $v$, denoted $\|v\|_2$ or simply $\|v\|$, is the straight-line distance from the origin $(0, \dots, 0)$ to the tip of $v$:
$$\|v\| = \sqrt{\sum_{i=1}^D v_i^2}$$

### 2.2 Dot Product (Scalar Product)
The dot product between two vectors $a$ and $b$ is the sum of the products of their corresponding components:
$$a \cdot b = \langle a, b \rangle = \sum_{i=1}^D a_i b_i$$

Geometric definition:
$$a \cdot b = \|a\| \|b\| \cos \theta$$
where $\theta$ is the angle between the two vectors in $D$-dimensional space.

### 2.3 Euclidean Distance ($L_2$ Distance)
The Euclidean distance measures the straight-line geometric distance between the endpoints of two vectors:
$$d_{\text{euclid}}(a, b) = \|a - b\|_2 = \sqrt{\sum_{i=1}^D (a_i - b_i)^2}$$

**Squared Euclidean Distance:**
$$d_{\text{euclid}}^2(a, b) = \sum_{i=1}^D (a_i - b_i)^2$$

### 2.4 Cosine Similarity
Cosine similarity evaluates the angular alignment between two non-zero vectors, completely independent of their magnitude:
$$\text{cosine\_sim}(a, b) = \cos \theta = \frac{a \cdot b}{\|a\| \|b\|} = \frac{\sum_{i=1}^D a_i b_i}{\sqrt{\sum_{i=1}^D a_i^2} \sqrt{\sum_{i=1}^D b_i^2}}$$

Properties:
- Range: $[-1.0, 1.0]$
- $\cos \theta = 1.0$: Exactly collinear (pointing in identical directions; $\theta = 0^\circ$).
- $\cos \theta = 0.0$: Orthogonal (perpendicular; $\theta = 90^\circ$, zero linear correlation).
- $\cos \theta = -1.0$: Diametrically opposite ($\theta = 180^\circ$).

---

## 3. Classroom Mathematical Example: Step-by-Step Proof

Consider the classroom coordinate points:
$$A = \begin{bmatrix} 2 \\ 3 \end{bmatrix}, \quad B = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad C = \begin{bmatrix} 0 \\ 4 \end{bmatrix}, \quad \text{Query } Q = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$$

### Step 1: Compute Vector Magnitudes
- $\|Q\| = \sqrt{2^2 + 4^2} = \sqrt{4 + 16} = \sqrt{20} \approx 4.472136$
- $\|A\| = \sqrt{2^2 + 3^2} = \sqrt{4 + 9} = \sqrt{13} \approx 3.605551$
- $\|B\| = \sqrt{1^2 + 2^2} = \sqrt{1 + 4} = \sqrt{5} \approx 2.236068$
- $\|C\| = \sqrt{0^2 + 4^2} = \sqrt{0 + 16} = \sqrt{16} = 4.000000$

### Step 2: Intermediate Calculations for Point A (2, 3)
1. **Coordinate differences $(Q - A)$:** $\begin{bmatrix} 2 - 2 \\ 4 - 3 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$
2. **Squared differences:** $0^2 = 0$, $1^2 = 1$
3. **Squared Euclidean distance:** $0 + 1 = 1$
4. **Euclidean distance:** $d(Q, A) = \sqrt{1} = \mathbf{1.000000}$
5. **Dot product:** $Q \cdot A = (2 \times 2) + (4 \times 3) = 4 + 12 = \mathbf{16}$
6. **Cosine similarity:**
   $$\text{cosine\_sim}(Q, A) = \frac{16}{\sqrt{20} \times \sqrt{13}} = \frac{16}{\sqrt{260}} = \frac{16}{16.124515} \approx \mathbf{0.992278}$$

### Step 3: Intermediate Calculations for Point B (1, 2)
1. **Coordinate differences $(Q - B)$:** $\begin{bmatrix} 2 - 1 \\ 4 - 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$
2. **Squared differences:** $1^2 = 1$, $2^2 = 4$
3. **Squared Euclidean distance:** $1 + 4 = 5$
4. **Euclidean distance:** $d(Q, B) = \sqrt{5} \approx \mathbf{2.236068}$
5. **Dot product:** $Q \cdot B = (2 \times 1) + (4 \times 2) = 2 + 8 = \mathbf{10}$
6. **Cosine similarity:**
   $$\text{cosine\_sim}(Q, B) = \frac{10}{\sqrt{20} \times \sqrt{5}} = \frac{10}{\sqrt{100}} = \frac{10}{10} = \mathbf{1.000000}$$

### Step 4: Intermediate Calculations for Point C (0, 4)
1. **Coordinate differences $(Q - C)$:** $\begin{bmatrix} 2 - 0 \\ 4 - 4 \end{bmatrix} = \begin{bmatrix} 2 \\ 0 \end{bmatrix}$
2. **Squared differences:** $2^2 = 4$, $0^2 = 0$
3. **Squared Euclidean distance:** $4 + 0 = 4$
4. **Euclidean distance:** $d(Q, C) = \sqrt{4} = \mathbf{2.000000}$
5. **Dot product:** $Q \cdot C = (2 \times 0) + (4 \times 4) = 0 + 16 = \mathbf{16}$
6. **Cosine similarity:**
   $$\text{cosine\_sim}(Q, C) = \frac{16}{\sqrt{20} \times 4} = \frac{16}{4\sqrt{20}} = \frac{4}{\sqrt{20}} = \frac{4}{4.472136} \approx \mathbf{0.894427}$$

---

## 4. Analysis: Why the Rankings Differ

| Metric | Ranking Order | Winner (Rank 1) | Loser (Rank 3) |
|---|---|---|---|
| **Euclidean Distance (Ascending)** | **$A \to C \to B$** | Point A ($d = 1.000000$) | Point B ($d = 2.236068$) |
| **Cosine Similarity (Descending)** | **$B \to A \to C$** | Point B ($\cos = 1.000000$) | Point C ($\cos = 0.894427$) |

### The Mathematical Reason:
1. **Euclidean Distance** measures the physical translation distance between two points in coordinate space. Point A is physically closest to Q because it only differs by 1 unit along the y-axis. Point B is physically furthest away ($d = \sqrt{5} \approx 2.236$).
2. **Cosine Similarity** measures the angle between vectors, disregarding length.
   Notice that $Q = \begin{bmatrix} 2 \\ 4 \end{bmatrix} = 2 \times \begin{bmatrix} 1 \\ 2 \end{bmatrix} = 2 \times B$.
   $Q$ is an exact scalar multiple of $B$. Therefore, $Q$ and $B$ point along the **exact same ray from the origin**. The angle $\theta$ between them is $0^\circ$, and $\cos(0^\circ) = 1.000000$.
   Hence, Point B receives a perfect score of 1.0, even though it is the furthest in Euclidean distance!

---

## 5. Normalization Proof: When Cosine Equals Dot Product

### 5.1 $L_2$ Normalization
An arbitrary non-zero vector $v$ is projected onto the unit hypersphere $\mathbb{S}^{D-1}$ by dividing by its Euclidean norm:
$$\hat{v} = \frac{v}{\|v\|_2}$$

Notice that:
$$\|\hat{v}\|_2 = \left\| \frac{v}{\|v\|_2} \right\|_2 = \frac{\|v\|_2}{\|v\|_2} = 1.0$$

### 5.2 Cosine Similarity on Normalized Vectors
$$\text{cosine\_sim}(\hat{a}, \hat{b}) = \frac{\hat{a} \cdot \hat{b}}{\|\hat{a}\| \|\hat{b}\|} = \frac{\hat{a} \cdot \hat{b}}{1 \times 1} = \hat{a} \cdot \hat{b}$$

**Fundamental Theorem:** The cosine similarity between two arbitrary vectors $a$ and $b$ is mathematically identical to the simple dot product of their $L_2$-normalized counterparts $\hat{a}$ and $\hat{b}$:
$$\text{cosine\_sim}(a, b) = \hat{a} \cdot \hat{b}$$

### 5.3 Relationship Between Euclidean Distance and Cosine Similarity on Unit Vectors
For any two unit vectors $\hat{a}, \hat{b} \in \mathbb{S}^{D-1}$:
$$\|\hat{a} - \hat{b}\|^2 = (\hat{a} - \hat{b}) \cdot (\hat{a} - \hat{b}) = \|\hat{a}\|^2 + \|\hat{b}\|^2 - 2(\hat{a} \cdot \hat{b})$$
Since $\|\hat{a}\|^2 = 1$ and $\|\hat{b}\|^2 = 1$:
$$\|\hat{a} - \hat{b}\|^2 = 1 + 1 - 2(\hat{a} \cdot \hat{b}) = 2 - 2 \cos(\theta)$$
Taking the square root:
$$d_{\text{euclid}}(\hat{a}, \hat{b}) = \sqrt{2 - 2 \cdot \text{cosine\_sim}(a, b)}$$

**Conclusion:** Because $f(x) = \sqrt{2 - 2x}$ is a strictly monotonically decreasing function for $x \in [-1, 1]$, **minimizing Euclidean distance on unit-normalized vectors produces the EXACT same ranking as maximizing Cosine similarity.**
