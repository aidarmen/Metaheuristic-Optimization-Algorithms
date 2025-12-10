# PSO Presentation with Slidev

This directory contains a Slidev presentation about Particle Swarm Optimization.

## Installation

1. Install Node.js (v16 or higher)

2. Navigate to the presentation directory:
```bash
cd presentation
```

3. Install dependencies:
```bash
npm install
```

## Running the Presentation

Start the development server:
```bash
npm run dev
```

Or from the project root:
```bash
cd presentation
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

This will create `slides-export.pdf` in the presentation folder.

## Export to PowerPoint

To export to PowerPoint (PPTX):

1. First export slides as PNG images:
```bash
npm run export -- --format png --output slides-export
```

2. Then use PowerPoint or other tools to create PPTX from the PNG images in `slides-export/` folder.

## Presentation Contents

The presentation includes:

1. **Introduction to PSO** - What is PSO and its advantages
2. **Algorithm Explanation** - Parameters and update equations
3. **Formula Breakdown** - Detailed explanation of velocity and position updates
4. **Visualizations** - Animated GIFs showing PSO in action:
   - Sphere Function
   - Rastrigin Function
   - Ackley Function
   - Beale Function
   - Goldstein-Price Function
   - Rotated Ellipsoid Function
5. **Convergence Analysis** - Convergence plots for different functions
6. **Comparisons** - PSO vs Traditional Methods (Gradient Descent, Hill Climbing)
7. **Adaptive PSO** - Next level PSO with adaptive parameters
8. **Real-World Applications** - Hyperparameter tuning, scheduling, resource allocation
9. **Key Takeaways** - Summary of PSO advantages

## File Structure

```
presentation/
├── slides.md              # Main presentation file
├── slides.config.ts       # Slidev configuration
├── package.json           # Dependencies and scripts
├── README_PRESENTATION.md # This file
├── slides-export/         # Exported PNG images (generated)
├── slides-export.pdf      # Exported PDF (generated)
└── slides.pptx           # Exported PowerPoint (generated)

../
├── media/                 # All images and GIFs
│   ├── *.png             # Convergence plots, comparisons
│   ├── *.gif             # PSO animations
│   └── *.jpg             # Formula diagrams
```

## Image Files

All image files (GIFs, PNGs, and JPGs) are stored in the `../media/` folder (parent directory). The presentation references them using paths like `../media/sphere_pso_animation.gif`.

The `slides.config.ts` is configured to use the parent directory as the public assets directory, so images should load correctly.

### If images are not loading:

1. Make sure all images are in the `../media/` folder
2. Check that `slides.config.ts` has `publicDir: '..'`
3. Verify image paths in `slides.md` start with `../media/`

## Navigation

- **Space** or **Arrow Right**: Next slide
- **Arrow Left**: Previous slide
- **F**: Fullscreen
- **O**: Overview mode
- **G**: Go to slide number

## Notes

- GIFs will animate in the browser presentation
- GIFs become static images in PDF/PPTX exports
- For animated GIFs, use the browser version (`npm run dev`)
