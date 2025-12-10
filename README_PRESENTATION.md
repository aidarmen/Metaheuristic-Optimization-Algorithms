# PSO Presentation with Slidev

This directory contains a Slidev presentation about Particle Swarm Optimization.

## Installation

1. Install Node.js (v16 or higher)

2. Install dependencies:
```bash
npm install
```

## Running the Presentation

Start the development server:
```bash
npm run dev
```

The presentation will open in your browser at `http://localhost:3030`

## Building

Build static files:
```bash
npm run build
```

Export as PDF:
```bash
npm run export
```

## Presentation Contents

The presentation includes:

1. **Introduction to PSO** - What is PSO and its advantages
2. **Algorithm Explanation** - Parameters and update equations
3. **Visualizations** - Animated GIFs showing PSO in action:
   - Sphere Function
   - Rastrigin Function
   - Ackley Function
   - Beale Function
   - Goldstein-Price Function
   - Rotated Ellipsoid Function
4. **Convergence Analysis** - Convergence plots for different functions
5. **Comparisons** - PSO vs Traditional Methods (Gradient Descent, Hill Climbing)
6. **Real-World Applications** - Hyperparameter tuning, scheduling, resource allocation
7. **Key Takeaways** - Summary of PSO advantages

## Image Files

All image files (GIFs and PNGs) should be in the **root directory** of the project. The presentation references them directly by filename (e.g., `sphere_pso_animation.gif`).

### If images are not loading:

**Option 1: Create a public folder (Recommended)**
```bash
mkdir public
# Copy all GIF and PNG files to public folder
copy *.gif public/
copy *.png public/
```

Then update `slides.md` to use paths like `/sphere_pso_animation.gif` (with leading slash).

**Option 2: Keep files in root**
Make sure all image files are in the same directory as `slides.md` and reference them without `./` prefix (already done in the presentation).

**Option 3: Use absolute paths**
If the above doesn't work, you can use full file paths or move images to a `public` folder and reference with `/filename.gif`.

## Navigation

- **Space** or **Arrow Right**: Next slide
- **Arrow Left**: Previous slide
- **F**: Fullscreen
- **O**: Overview mode
- **G**: Go to slide number

## Files

- `slides.md` - Main presentation file
- `package.json` - Dependencies and scripts
- All GIF and PNG files in the root directory are used in the presentation
