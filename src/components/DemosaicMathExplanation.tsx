import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { HelpTooltip } from '@/components/ui/HelpTooltip';
import { DemosaicAlgorithm, CFAType, DemosaicInput, ErrorStats, DemosaicParams } from '@/types/demosaic';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import { InteractiveDemosaicVisualizer } from './InteractiveDemosaicVisualizer';
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';
import { computeLaplacianMagnitude, normalizeScalarField, heatmapFromNormalizedField } from '@/lib/utils';

interface MathPanelProps {
  cfaType: CFAType;
  algorithm: DemosaicAlgorithm;
  x?: number;
  y?: number;
  error?: {
    rgb: { r: number, g: number, b: number };
    l2: number;
  };
  errorStats?: ErrorStats | null;
  input?: DemosaicInput | null;
  syntheticType?: string | null;
  params?: DemosaicParams;
}

interface ScalarFieldHeatmapProps {
  field: Float32Array | undefined;
  width: number;
  height: number;
  label: string;
}

function ScalarFieldHeatmap({ field, width, height, label }: ScalarFieldHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const lastMousePos = useRef({ x: 0, y: 0 });
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  
  // Keep refs in sync with state
  useEffect(() => {
    zoomRef.current = zoom;
    panRef.current = pan;
  }, [zoom, pan]);

  useEffect(() => {
    if (!field || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    const norm = normalizeScalarField(field, 0.01);
    const img = heatmapFromNormalizedField(norm, width, height);
    canvasRef.current.width = width;
    canvasRef.current.height = height;
    ctx.putImageData(img, 0, 0);
  }, [field, width, height]);

  // Set up native wheel event listener to avoid passive listener issues
  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    
    const handleWheel = (e: WheelEvent) => {
      // Allow page scrolling if not zooming significantly or at edges, but 
      // for consistency, let's zoom if inside the component.
      const currentZoom = zoomRef.current;
      const currentPan = panRef.current;
      const scaleFactor = 0.1;
      const delta = -Math.sign(e.deltaY) * scaleFactor;
      const newZoom = Math.max(1, Math.min(currentZoom + delta * currentZoom, 20));
      
      if (newZoom !== currentZoom && containerRef.current) {
        // Only prevent default if we are actually zooming (to avoid blocking scroll when at limit)
        e.preventDefault();
        
        const rect = containerRef.current.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const worldX = (mouseX - currentPan.x) / currentZoom;
        const worldY = (mouseY - currentPan.y) / currentZoom;

        const newPanX = mouseX - worldX * newZoom;
        const newPanY = mouseY - worldY * newZoom;

        setZoom(newZoom);
        setPan({ x: newPanX, y: newPanY });
      }
    };
    
    element.addEventListener('wheel', handleWheel, { passive: false });
    
    return () => {
      element.removeEventListener('wheel', handleWheel);
    };
  }, []); // Empty deps - handler reads from refs

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    lastMousePos.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    e.preventDefault();
    const dx = e.clientX - lastMousePos.current.x;
    const dy = e.clientY - lastMousePos.current.y;
    setPan(p => ({ x: p.x + dx, y: p.y + dy }));
    lastMousePos.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseUp = () => setIsDragging(false);
  const handleMouseLeave = () => setIsDragging(false);

  const handleZoomButton = (direction: 'in' | 'out') => {
     if (!containerRef.current) return;
     const rect = containerRef.current.getBoundingClientRect();
     const cx = rect.width / 2;
     const cy = rect.height / 2;
     const factor = direction === 'in' ? 1.2 : 1/1.2;
     const newZoom = Math.max(1, Math.min(20, zoom * factor));
     
     const worldX = (cx - pan.x) / zoom;
     const worldY = (cy - pan.y) / zoom;
     
     const newPanX = cx - worldX * newZoom;
     const newPanY = cy - worldY * newZoom;
     
     setZoom(newZoom);
     setPan({ x: newPanX, y: newPanY });
  };

  const resetZoom = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
        <div className="flex items-center gap-1 bg-muted/50 rounded px-1">
           <button onClick={() => handleZoomButton('out')} className="p-1 hover:text-primary rounded-sm hover:bg-background/50" title="Zoom Out"><ZoomOut className="w-3 h-3" /></button>
           <span className="text-[10px] w-8 text-center font-mono select-none">{Math.round(zoom * 100)}%</span>
           <button onClick={() => handleZoomButton('in')} className="p-1 hover:text-primary rounded-sm hover:bg-background/50" title="Zoom In"><ZoomIn className="w-3 h-3" /></button>
           <button onClick={resetZoom} className="p-1 hover:text-primary ml-1 rounded-sm hover:bg-background/50" title="Reset View"><RotateCcw className="w-3 h-3" /></button>
        </div>
      </div>
      <div 
        ref={containerRef}
        className={`border rounded overflow-hidden bg-background relative touch-none ${isDragging ? 'cursor-grabbing' : 'cursor-grab'}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
      >
        <canvas
          ref={canvasRef}
          className="w-full h-auto block origin-top-left transition-transform duration-75 ease-linear"
          style={{ 
             imageRendering: 'pixelated',
             transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`
          }}
        />
      </div>
    </div>
  );
}

// Define SYNTHETIC_EXPLANATIONS as a function to avoid initialization order issues
const getSyntheticExplanations = (): Record<string, { 
    title: string; 
    whatToLookFor: string;
    analysis: (cfa: CFAType, algo: DemosaicAlgorithm) => React.ReactNode;
}> => ({
    'zoneplate': {
        title: "Zone Plate",
        whatToLookFor: "Observe the center for clarity and the edges for color artifacts (Moiré). High-frequency details often cause false colors in simple demosaicing.",
        analysis: (cfa, algo) => (
            <>
                With <strong>{cfa.toUpperCase()}</strong> and <strong>{algo}</strong>, you likely see color rings where there should only be black and white. 
                {algo === 'nearest' && " Nearest Neighbor produces jagged, blocky artifacts."}
                {algo === 'bilinear' && " Bilinear interpolation blurs the high frequencies but still suffers from 'zippering' false colors."}
                {(algo === 'niu_edge_sensing' || algo === 'lien_edge_based') && " Edge-aware algorithms should show reduced false colors compared to bilinear, as they interpolate along edges rather than across them."}
                {algo === 'wu_polynomial' && " Polynomial interpolation provides better frequency response, reducing aliasing artifacts while maintaining edge sharpness."}
                {algo === 'kiku_residual' && " Residual interpolation adapts to local structures, showing improved artifact reduction especially in high-frequency regions."}
            </>
        )
    },
    'checker': {
        title: "Fine Checkerboard",
        whatToLookFor: "This pattern represents the Nyquist limit (max frequency). Ideally, it should look like gray or distinct pixels, but often turns into solid colors.",
        analysis: (cfa, algo) => (
            <>
                A 1-pixel checkerboard is the worst-case scenario for <strong>{cfa.toUpperCase()}</strong>. 
                {algo === 'nearest' && " It completely breaks the structure, creating large solid color blocks."}
                {algo === 'bilinear' && " It averages out to a muddy gray, losing all detail."}
                {(algo === 'niu_edge_sensing' || algo === 'lien_edge_based' || algo === 'wu_polynomial' || algo === 'kiku_residual') && " Advanced algorithms may preserve some structure better than bilinear, but this pattern remains extremely challenging due to its frequency being at the Nyquist limit."}
            </>
        )
    },
    'sweep': {
        title: "Color Sweep",
        whatToLookFor: "Smooth gradients should remain smooth. Banding or jagged transitions indicate interpolation errors.",
        analysis: (cfa, algo) => (
            <>
                <strong>{algo}</strong> demosaicing {algo === 'nearest' ? "struggles with smooth transitions, creating 'steps' in the gradient." : algo === 'bilinear' ? "generally handles gradients well, but may desaturate high-frequency color borders." : "should handle smooth gradients well, with edge-aware methods preserving transitions more accurately than simple bilinear interpolation."}
            </>
        )
    },
    'star': {
        title: "Starburst",
        whatToLookFor: "Lines should remain straight and crisp. Look for 'stair-stepping' (aliasing) on diagonal lines.",
        analysis: (cfa, algo) => (
            <>
                Diagonal lines are challenging on a square <strong>{cfa.toUpperCase()}</strong> grid. 
                {algo === 'nearest' && " You will see significant jagged edges (aliasing)."}
                {algo === 'bilinear' && " The lines will be smoother but slightly blurred."}
                {(algo === 'niu_edge_sensing' || algo === 'lien_edge_based') && " Edge-aware algorithms should better preserve diagonal edges by detecting edge direction and interpolating accordingly."}
                {algo === 'wu_polynomial' && " Polynomial interpolation may provide sharper diagonal lines with reduced aliasing."}
                {algo === 'kiku_residual' && " Residual interpolation adapts to diagonal structures, potentially reducing artifacts along the lines."}
            </>
        )
    },
    'diagonal': {
        title: "Diagonal Lines",
        whatToLookFor: "Look for 'zippering' artifacts along diagonal lines. Bayer patterns are particularly sensitive to diagonal aliasing, which can create false colors.",
        analysis: (cfa, algo) => (
            <>
                Diagonal lines at 45° angles are the worst-case scenario for <strong>{cfa.toUpperCase()}</strong> patterns. 
                {algo === 'nearest' && " You'll see severe zippering (alternating color bands) along the lines."}
                {algo === 'bilinear' && " The zippering is reduced but still visible, especially with high-contrast lines."}
                {(algo === 'niu_edge_sensing' || algo === 'lien_edge_based') && " Edge-aware methods should significantly reduce zippering by detecting diagonal edges and interpolating along them."}
                {algo === 'wu_polynomial' && " Polynomial interpolation may reduce zippering artifacts through better local structure modeling."}
                {algo === 'kiku_residual' && " Residual interpolation adapts to diagonal structures, showing improved zippering reduction."}
                The square grid structure of CFA patterns doesn't align well with diagonal features, but advanced algorithms mitigate this.
            </>
        )
    },
    'sine': {
        title: "Sine Wave Gratings",
        whatToLookFor: "Observe how different frequencies are handled. High frequencies near Nyquist should show aliasing and false colors. The three regions test horizontal, vertical, and diagonal orientations.",
        analysis: (cfa, algo) => (
            <>
                Sine gratings test the frequency response of <strong>{algo}</strong> demosaicing. 
                {algo === 'nearest' && " You'll see strong aliasing at high frequencies, with false colors appearing where there should only be grayscale."}
                {algo === 'bilinear' && " The algorithm acts as a low-pass filter, blurring high frequencies but reducing aliasing artifacts."}
                {(algo === 'niu_edge_sensing' || algo === 'lien_edge_based') && " Edge-aware algorithms should show better frequency response with reduced false colors, especially for horizontal and vertical gratings."}
                {algo === 'wu_polynomial' && " Polynomial interpolation provides superior frequency response, reducing aliasing while maintaining signal fidelity better than bilinear."}
                {algo === 'kiku_residual' && " Residual interpolation adapts to frequency content, showing improved handling of high-frequency components."}
                Diagonal gratings typically show the worst artifacts due to the square grid structure.
            </>
        )
    },
    'patches': {
        title: "Color Patches",
        whatToLookFor: "Pure color squares should remain saturated and accurate. Look for color bleeding at edges, desaturation, or incorrect color reproduction.",
        analysis: (cfa, algo) => (
            <>
                Color patches test color accuracy and saturation preservation. 
                {algo === 'nearest' && " You may see color bleeding at patch boundaries and slight desaturation due to incorrect neighbor selection."}
                {algo === 'bilinear' && " Colors should be more accurate, but edges between patches may show color fringing or desaturation."}
                {(algo === 'niu_edge_sensing' || algo === 'lien_edge_based') && " Edge-aware methods should preserve color saturation better at patch boundaries by avoiding interpolation across edges."}
                {algo === 'wu_polynomial' && " Polynomial interpolation may provide more accurate color reproduction with better saturation preservation."}
                {algo === 'kiku_residual' && " Residual interpolation adapts to color boundaries, showing improved color accuracy and reduced fringing."}
                The <strong>{cfa.toUpperCase()}</strong> pattern's color distribution affects how well each primary color is captured.
            </>
        )
    },
    'fringes': {
        title: "Color Fringes",
        whatToLookFor: "Thin colored lines on neutral backgrounds should remain crisp. Look for color bleeding, false colors spreading beyond the lines, or desaturation of the colored lines.",
        analysis: (cfa, algo) => (
            <>
                Color fringes are a common real-world scenario (e.g., chromatic aberration). 
                {algo === 'nearest' && " You'll see significant color bleeding, with false colors spreading into the neutral gray background."}
                {algo === 'bilinear' && " The bleeding is reduced, but thin lines may appear slightly desaturated or blurred."}
                {(algo === 'niu_edge_sensing' || algo === 'lien_edge_based') && " Edge-aware algorithms should better preserve thin colored lines by detecting edges and avoiding interpolation that would cause bleeding."}
                {algo === 'wu_polynomial' && " Polynomial interpolation may preserve thin lines more accurately with reduced bleeding into the background."}
                {algo === 'kiku_residual' && " Residual interpolation adapts to the high-frequency color boundaries, showing improved preservation of thin colored lines."}
                This pattern is particularly challenging because it combines high spatial frequency (thin lines) with pure colors.
            </>
        )
    }
});

export function DemosaicMathExplanation({ 
  cfaType, 
  algorithm, 
  x,
  y,
  error,
  errorStats,
  input,
  syntheticType,
  params
}: MathPanelProps) {
  const syntheticExplanations = useMemo(() => getSyntheticExplanations(), []);
  
  const laplacianField = useMemo(() => {
    if (!input?.groundTruthRGB) return undefined;
    try {
      return computeLaplacianMagnitude(input.groundTruthRGB);
    } catch {
      return undefined;
    }
  }, [input?.groundTruthRGB]);

  return (
    <Card className="h-full border-l border-border shadow-sm rounded-none lg:rounded-lg flex flex-col bg-card">
      <CardHeader className="pb-2 shrink-0">
        <CardTitle className="text-xl font-semibold text-primary flex items-center gap-2">
          Inspection & Math
          <HelpTooltip content="This panel shows the inner workings of the demosaicing process. Click a pixel on the canvas to see a step-by-step trace of how its color was calculated from the raw sensor data." />
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden flex flex-col px-3 sm:px-4">
        <Tabs defaultValue="visualizer" className="w-full flex flex-col h-full">
          <TabsList className="grid w-full grid-cols-2 md:grid-cols-3 h-auto mb-4 shrink-0">
            <TabsTrigger value="visualizer">Visualizer</TabsTrigger>
            <TabsTrigger value="algo">Algorithm Info</TabsTrigger>
            <TabsTrigger value="error">Error Analysis</TabsTrigger>
          </TabsList>
          
          <TabsContent value="visualizer" className="space-y-4 animate-in fade-in-50 overflow-y-auto pr-1 flex-1">
             {input && x !== undefined && y !== undefined && x >= 0 ? (
                 <InteractiveDemosaicVisualizer 
                    input={input}
                    centerX={x}
                    centerY={y}
                    algorithm={algorithm}
                    params={params}
                 />
             ) : (
                <div className="flex flex-col items-center justify-center h-[300px] text-center p-6 border-2 border-dashed border-muted rounded-lg bg-muted/10">
                    <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
                      <ZoomIn className="w-6 h-6 text-muted-foreground" />
                    </div>
                    <h4 className="font-semibold text-foreground">Click to Visualize</h4>
                    <p className="text-sm text-muted-foreground mt-1">
                      Click a pixel on the image to see the zoomed-in convolution process.
                    </p>
                </div>
             )}
          </TabsContent>

          <TabsContent value="algo" className="space-y-4 animate-in fade-in-50 overflow-y-auto pr-1">
            <div className="text-sm space-y-4">
              
              {syntheticType && syntheticExplanations[syntheticType] && (
                <div className="bg-primary/5 border border-primary/20 p-3 rounded-md space-y-2">
                  <div className="flex items-center gap-2">
                    <h3 className="font-bold text-primary text-base">{syntheticExplanations[syntheticType].title} Analysis</h3>
                  </div>
                  
                  <div className="space-y-3">
                    <div>
                        <h4 className="text-xs font-bold uppercase text-muted-foreground mb-1">What to Look For</h4>
                        <p className="text-xs text-foreground leading-relaxed">
                            {syntheticExplanations[syntheticType].whatToLookFor}
                        </p>
                    </div>
                    
                    <div className="pt-2 border-t border-primary/10">
                        <h4 className="text-xs font-bold uppercase text-muted-foreground mb-1">Algorithm Performance</h4>
                        <div className="text-xs text-foreground leading-relaxed">
                            {syntheticExplanations[syntheticType].analysis(cfaType, algorithm)}
                        </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="bg-muted/30 p-3 rounded-md space-y-2">
                <h3 className="font-semibold text-primary">{cfaType.toUpperCase()} Pattern</h3>
                <div className="text-muted-foreground text-xs leading-relaxed space-y-2">
                  {cfaType === 'bayer' && (
                    <>
                      <p>The Bayer filter mosaic arranges RGB color filters on a square grid. The sampling function <InlineMath math="S(x,y)" /> selects one color channel per pixel:</p>
                      <BlockMath math="S(x,y) \in \{R, G, B\}" />
                      <p>Typically 50% Green, 25% Red, 25% Blue to mimic human eye sensitivity.</p>
                    </>
                  )}
                  {cfaType === 'xtrans' && (
                    <>
                      <p>Fujifilm X-Trans uses a <InlineMath math="6 \times 6" /> pseudo-random pattern. This periodicity reduces Moiré patterns, eliminating the need for an optical low-pass filter.</p>
                      <p>Every row and column contains all three colors (R, G, B).</p>
                    </>
                  )}
                  {cfaType === 'foveon' && "Foveon sensors stack three photodiodes vertically, capturing R, G, and B at every pixel location directly. No spatial demosaicing is required in the traditional sense."}
                </div>
              </div>

              <div className="bg-muted/30 p-3 rounded-md space-y-2">
                 <h3 className="font-semibold text-primary">Algorithm: {algorithm}</h3>
                 <div className="text-muted-foreground text-xs leading-relaxed space-y-2">
                    {algorithm === 'nearest' && (
                      <>
                        <p>Copies values from the nearest available neighbor of the missing color.</p>
                        <BlockMath math="\hat{C}(x,y) = C(x', y')" />
                        <p>Where <InlineMath math="(x', y')" /> is the coordinate of the closest pixel with color <InlineMath math="C" />. This is 0th-order interpolation, causing "zippering" artifacts.</p>
                      </>
                    )}
                    {algorithm === 'bilinear' && (
                      <>
                        <p>Computes the arithmetic mean of immediate neighbors of the same color.</p>
                        <p>For Green at a Red/Blue location:</p>
                        <BlockMath math="\hat{G}(x,y) = \frac{1}{4} \sum_{(i,j) \in \mathcal{N}_4} G(x+i, y+j)" />
                        <p>Where <InlineMath math="\mathcal{N}_4" /> are the 4 cardinal neighbors (top, bottom, left, right). This acts as a low-pass filter.</p>
                      </>
                    )}
                    {algorithm === 'malvar' && (
                      <>
                        <p><strong>High-Quality Linear Interpolation</strong> (Malvar, He, Cutler). It adds a gradient correction term to the bilinear estimate to preserve edges.</p>
                        <p>For example, estimating Green at Red pixel:</p>
                        <BlockMath math="\hat{G}(x,y) = \hat{G}_{bilinear}(x,y) + \alpha \Delta_{R}" />
                        <p>Where <InlineMath math="\Delta_R" /> is the Laplacian of the Red channel (2nd derivative), effectively using the Red channel's high-frequency detail to guide the Green interpolation.</p>
                      </>
                    )}
                    {algorithm === 'niu_edge_sensing' && (
                      <>
                        <p><strong>Low-Cost Edge Sensing</strong>. This algorithm improves upon the Hamilton-Adams method by introducing a low-cost edge sensing scheme that guides interpolation using directional variations.</p>
                        <p>The algorithm computes directional variations in horizontal, vertical, and diagonal directions:</p>
                        <BlockMath math="\Delta_H = |I(x+1,y) - I(x-1,y)|, \quad \Delta_V = |I(x,y+1) - I(x,y-1)|" />
                        <p>These variations are then weighted using a logistic function to determine edge strength:</p>
                        <BlockMath math="w = \frac{1}{1 + e^{-k(\Delta - \theta)}}" />
                        <p>Where <InlineMath math="k" /> controls steepness and <InlineMath math="\theta" /> is the edge detection threshold. The algorithm interpolates along edges (where variation is low) rather than across them, preserving sharp boundaries while maintaining computational efficiency.</p>
                        <p><strong>Key advantage:</strong> Achieves high accuracy with low computational cost, processing high-resolution images much faster than top-performing methods while maintaining comparable quality.</p>
                      </>
                    )}
                    {algorithm === 'lien_edge_based' && (
                      <>
                        <p><strong>Hamilton-Adams (Edge-Based)</strong>. This method uses simple operations (addition, subtraction, shift, comparison) to detect edges and guide interpolation, making it highly suitable for hardware implementation.</p>
                        <p>The algorithm detects edge direction by comparing color differences:</p>
                        <BlockMath math="\text{edge} = \begin{cases} \text{horizontal} & \text{if } |I(x-1,y) - I(x+1,y)| < |I(x,y-1) - I(x,y+1)| \\ \text{vertical} & \text{otherwise} \end{cases}" />
                        <p>Interpolation is then performed along the detected edge direction to preserve structural details. For example, if a horizontal edge is detected, the algorithm interpolates vertically (along the edge) rather than horizontally (across the edge).</p>
                        <p><strong>Key advantage:</strong> Designed for efficient VLSI implementation with minimal line buffering (only 4 lines), enabling real-time processing at high throughput rates (approximately 200 million samples per second).</p>
                      </>
                    )}
                    {algorithm === 'wu_polynomial' && (
                      <>
                        <p><strong>Polynomial Interpolation</strong>. This algorithm uses polynomial interpolation instead of traditional bilinear or Laplacian predictors, providing more accurate estimation of missing color values.</p>
                        <p>The method introduces polynomial error predictors that better capture local image structure. For a set of neighboring values, the algorithm fits a polynomial:</p>
                        <BlockMath math="P_n(x) = \sum_{i=0}^{n} a_i x^i" />
                        <p>Where <InlineMath math="n" /> is the polynomial degree (typically 2-3). The algorithm also classifies edges using color differences to guide the interpolation process, then applies a weighted sum strategy in a refinement stage to reduce artifacts.</p>
                        <p><strong>Key advantage:</strong> More accurate than traditional interpolation methods, particularly effective for mobile devices and tablets where both quality and computational efficiency are important.</p>
                      </>
                    )}
                    {algorithm === 'kiku_residual' && (
                      <>
                        <p><strong>Residual Interpolation</strong>. Rather than directly interpolating missing colors, the algorithm interpolates <em>residuals</em> (what the initial guess got wrong) and adds those corrections back.</p>
                        <ol className="list-decimal list-inside space-y-1 text-xs">
                          <li>
                            <strong>Baseline:</strong> Build an initial estimate with bilinear interpolation <InlineMath math="\hat{I}_{0,c}(x,y)" /> for each channel <InlineMath math="c \in \{R,G,B\}" />.
                          </li>
                          <li>
                            <strong>Residuals at measured samples:</strong> For pixels that actually observe channel <InlineMath math="c" />, compute the error
                            <BlockMath math="R_c(x,y) = I^{\text{obs}}_c(x,y) - \hat{I}_{0,c}(x,y)" />
                            (zero elsewhere).
                          </li>
                          <li>
                            <strong>Interpolate the residual field:</strong> Spread those residuals to missing locations with a neighborhood average over same-color samples
                            <BlockMath math="\tilde{R}_c(x,y) = \frac{1}{|\mathcal{N}_c|} \sum_{(i,j)\in \mathcal{N}_c} R_c(x+i,y+j)" />
                          </li>
                          <li>
                            <strong>Refine:</strong> Add the interpolated residuals back: <InlineMath math="\hat{I}_c = \hat{I}_{0,c} + \tilde{R}_c" />. Repeat for a small number of iterations if desired.
                          </li>
                        </ol>
                        <p><strong>Why it helps:</strong> The baseline handles smooth areas; the residual field re-injects high-frequency structure and cross-channel correlations, reducing zippering and false colors compared with pure color-difference interpolation.</p>
                      </>
                    )}
                 </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="error" className="space-y-4 animate-in fade-in-50 overflow-y-auto pr-1 flex-1">
            {input && input.groundTruthRGB && errorStats ? (
              <div className="space-y-4 pb-4 text-xs">
                <div className="bg-muted/40 p-3 rounded-md border border-border/60 space-y-2">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-primary text-sm">Global Error Metrics</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center gap-1">
                        <div className="text-[10px] uppercase tracking-wide text-muted-foreground">PSNR (dB)</div>
                        <HelpTooltip className="h-3 w-3" content={
                          <div className="space-y-2">
                            <div>
                              <strong>Peak Signal-to-Noise Ratio (PSNR)</strong>
                            </div>
                            <div>Higher is better. Measures quality of reconstruction.</div>
                            <div className="pt-1 border-t border-yellow-200">
                              <div className="font-semibold mb-1">Interpretation:</div>
                              <div className="space-y-0.5 text-xs">
                                <div>• <strong>&gt; 40 dB:</strong> Excellent — artifacts barely visible</div>
                                <div>• <strong>30-40 dB:</strong> Good — minor artifacts</div>
                                <div>• <strong>20-30 dB:</strong> Fair — noticeable artifacts</div>
                                <div>• <strong>&lt; 20 dB:</strong> Poor — severe artifacts</div>
                              </div>
                            </div>
                          </div>
                        } />
                      </div>
                      <div className="font-mono text-xs">
                        <div>Total: {errorStats.psnr?.total?.toFixed(2) ?? 'N/A'}</div>
                        <div className="text-muted-foreground">
                          R: {errorStats.psnr?.r?.toFixed(2) ?? 'N/A'}, G: {errorStats.psnr?.g?.toFixed(2) ?? 'N/A'}, B: {errorStats.psnr?.b?.toFixed(2) ?? 'N/A'}
                        </div>
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">MSE / MAE</div>
                      <div className="font-mono text-xs">
                        <div>MSE: {errorStats.mse?.total?.toFixed(2) ?? 'N/A'}</div>
                        <div>MAE: {errorStats.mae?.total?.toFixed(2) ?? 'N/A'}</div>
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">SSIM</div>
                      <div className="font-mono text-xs">{errorStats.ssim?.toFixed(4) ?? 'N/A'}</div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
                        Pixel ({x ?? '-'}, {y ?? '-'})
                      </div>
                      {error && error.l2 !== undefined ? (
                        <div className="font-mono text-xs">
                          L2:{' '}
                          <span className={error.l2 > 0.1 ? 'text-red-500' : 'text-green-600'}>
                            {error.l2.toFixed(4)}
                          </span>
                        </div>
                      ) : (
                        <div className="text-muted-foreground text-[11px]">
                          Click a pixel on the image to see local error.
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-muted/40 p-3 rounded-md border border-border/60 space-y-2">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold text-primary text-sm">Error Heatmap (‖error‖₂)</h3>
                    </div>
                    <p className="text-[11px] text-muted-foreground leading-relaxed">
                      Each pixel is colored by the L2 distance between reconstructed and ground-truth RGB. 
                      Blue means low error, red means high error. Notice how errors cluster around edges and fine detail.
                    </p>
                    {errorStats.l2Map && (
                      <ScalarFieldHeatmap
                        field={errorStats.l2Map}
                        width={input.groundTruthRGB.width}
                        height={input.groundTruthRGB.height}
                        label="‖RGB_est - RGB_true‖₂"
                      />
                    )}
                  </div>
                  <div className="bg-muted/40 p-3 rounded-md border border-border/60 space-y-2">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold text-primary text-sm">Laplacian Magnitude (∇²I)</h3>
                    </div>
                    <p className="text-[11px] text-muted-foreground leading-relaxed">
                      This map approximates the second spatial derivative of the ground-truth luminance. 
                      Large values correspond to strong curvature (sharp edges, oscillations). 
                      For linear interpolation, the demosaicing error can be bounded by a constant times this curvature term.
                    </p>
                    {laplacianField && (
                      <ScalarFieldHeatmap
                        field={laplacianField}
                        width={input.groundTruthRGB.width}
                        height={input.groundTruthRGB.height}
                        label="‖∇²I‖"
                      />
                    )}
                  </div>
                </div>

                <div className="bg-muted/30 p-3 rounded-md space-y-3">
                  <h3 className="font-semibold text-primary text-sm">Theoretical Error Analysis</h3>
                  
                  <div className="space-y-3">
                    <div>
                      <p className="text-[11px] text-muted-foreground leading-relaxed mb-2">
                        We analyze reconstruction error <InlineMath math="e_c(x,y) := I_c^\text{CFA}(x,y) - \hat{I}_c(x,y)" />, 
                        where <InlineMath math="I_c^\text{CFA}" /> is the CFA-sampled value and <InlineMath math="\hat{I}_c" /> 
                        is the algorithm's reconstruction. We assume smoothness:
                      </p>
                      <BlockMath math="|\nabla I_c|_\infty \le L, \qquad |H I_c|_\infty \le M" />
                      <p className="text-[11px] text-muted-foreground leading-relaxed mt-2">
                        where <InlineMath math="L" /> bounds gradients and <InlineMath math="M" /> bounds curvature (Hessian).
                      </p>
                    </div>

                    {algorithm === 'nearest' && (
                      <div>
                        <h4 className="text-xs font-semibold text-primary mb-1">Nearest Neighbor</h4>
                        <p className="text-[11px] text-muted-foreground leading-relaxed">
                          Zero-order hold: error bounded by first derivative. For spacing <InlineMath math="\Delta x" />:
                        </p>
                        <BlockMath math="|f(x) - f(x_0)| \le \Delta x \, \max_{x_0 \le t \le x_0 + \Delta x} |f'(t)|" />
                      </div>
                    )}

                    {algorithm === 'bilinear' && (
                      <div>
                        <h4 className="text-xs font-semibold text-primary mb-1">Bilinear Interpolation</h4>
                        <p className="text-[11px] text-muted-foreground leading-relaxed mb-2">
                          <strong>Worst-case:</strong> For 8-bit images, <InlineMath math="|I_c - \hat{I}_c| \le 255" /> per channel (trivial bound).
                        </p>
                        <p className="text-[11px] text-muted-foreground leading-relaxed mb-2">
                          <strong>Smooth images:</strong> For Bayer CFA with neighbor distance <InlineMath math="d = \sqrt{2}" />:
                        </p>
                        <BlockMath math="|I_c(x,y) - \hat{I}_c(x,y)| \le \frac{M d^2}{2} = M" />
                        <BlockMath math="PSNR_c \ge 20\log_{10}\left(\frac{255}{M}\right)" />
                        <p className="text-[11px] text-muted-foreground leading-relaxed mt-2">
                          Examples: <InlineMath math="M \approx 1" /> (smooth sky) → PSNR ≥ 48 dB; 
                          <InlineMath math="M \approx 10" /> (gentle texture) → PSNR ≥ 28 dB; 
                          <InlineMath math="M \approx 50" /> (moderate edges) → PSNR ≥ 14 dB.
                        </p>
                      </div>
                    )}

                    {algorithm === 'niu_edge_sensing' && (
                      <div>
                        <h4 className="text-xs font-semibold text-primary mb-1">Niu Edge-Sensing</h4>
                        <p className="text-[11px] text-muted-foreground leading-relaxed mb-2">
                          Uses asymmetric directional weights. When edge detection works well (normalized max weight ≈ 0.5):
                        </p>
                        <BlockMath math="|e_c| \lesssim M + 0.5L\sqrt{2}" />
                        <BlockMath math="PSNR_c \ge 20\log_{10}\frac{255}{M + 0.5L\sqrt{2}}" />
                        <p className="text-[11px] text-muted-foreground leading-relaxed mt-2">
                          For smooth sky (<InlineMath math="M \approx 1, L \approx 5" />): error ≤ 4.5, PSNR ≳ 35 dB. 
                          When edge detection fails, performance degrades toward bilinear.
                        </p>
                      </div>
                    )}

                    {algorithm === 'lien_edge_based' && (
                      <div>
                        <h4 className="text-xs font-semibold text-primary mb-1">Hamilton-Adams Edge-Based</h4>
                        <p className="text-[11px] text-muted-foreground leading-relaxed mb-2">
                          Interpolates perpendicular to detected edges. When correct (distance <InlineMath math="d = 1" />):
                        </p>
                        <BlockMath math="|C(x,y) - \hat{C}(x,y)| \lesssim \frac{M}{2}" />
                        <BlockMath math="PSNR_c \ge 20\log_{10}\frac{510}{M}" />
                        <p className="text-[11px] text-muted-foreground leading-relaxed mt-2">
                          Examples: <InlineMath math="M \approx 1" /> → PSNR ≳ 54 dB; 
                          <InlineMath math="M \approx 10" /> → PSNR ≳ 34 dB. 
                          Misclassification degrades to bilinear-like error (<InlineMath math="\sim M" />).
                        </p>
                      </div>
                    )}

                    {algorithm === 'wu_polynomial' && (
                      <div>
                        <h4 className="text-xs font-semibold text-primary mb-1">Wu Polynomial Interpolation</h4>
                        <p className="text-[11px] text-muted-foreground leading-relaxed mb-2">
                          Uses distance-weighted averaging. For symmetric neighbors at <InlineMath math="d = \sqrt{2}" />:
                        </p>
                        <BlockMath math="|C(x,y) - \hat{C}(x,y)| \lesssim M" />
                        <BlockMath math="PSNR_c \ge 20\log_{10}\frac{255}{M}" />
                        <p className="text-[11px] text-muted-foreground leading-relaxed mt-2">
                          Shares the same worst-case bound as bilinear. Distance weighting improves empirical performance by 
                          downweighting farther neighbors but doesn't tighten the uniform curvature bound.
                        </p>
                      </div>
                    )}

                    {algorithm === 'kiku_residual' && (
                      <div>
                        <h4 className="text-xs font-semibold text-primary mb-1">Kiku Residual Interpolation</h4>
                        <p className="text-[11px] text-muted-foreground leading-relaxed mb-2">
                          Iteratively refines by interpolating residuals. With geometric decay <InlineMath math="\alpha \approx 0.6" /> 
                          and <InlineMath math="K = 3" /> iterations:
                        </p>
                        <BlockMath math="|e_c| \lesssim 0.22M, \qquad PSNR_c \gtrsim 20\log_{10}\frac{1159}{M}" />
                        <p className="text-[11px] text-muted-foreground leading-relaxed mt-2">
                          For <InlineMath math="M \approx 10" />: error ≤ 2.2, PSNR ≳ 41 dB. This is empirical and assumes 
                          residual smoothness, not a strict worst-case guarantee.
                        </p>
                      </div>
                    )}

                    {(algorithm === 'malvar' || (algorithm !== 'nearest' && algorithm !== 'bilinear' && algorithm !== 'niu_edge_sensing' && algorithm !== 'lien_edge_based' && algorithm !== 'wu_polynomial' && algorithm !== 'kiku_residual')) && (
                      <div>
                        <h4 className="text-xs font-semibold text-primary mb-1">General Linear Methods</h4>
                        <p className="text-[11px] text-muted-foreground leading-relaxed mb-2">
                          For linear interpolation methods, Taylor expansion shows the linear term cancels, leaving curvature error:
                        </p>
                        <BlockMath math="|I_c(x,y) - \hat{I}_c(x,y)| \le \frac{M d^2}{2}" />
                        <p className="text-[11px] text-muted-foreground leading-relaxed mt-2">
                          where <InlineMath math="d" /> is the maximum distance to neighbors used. This explains why high-frequency 
                          textures produce larger artifacts than smooth regions.
                        </p>
                      </div>
                    )}

                    <div className="pt-2 border-t border-border/50">
                      <h4 className="text-xs font-semibold text-primary mb-2">Comparison (M=10, L=50)</h4>
                      <div className="text-[10px] space-y-1 font-mono">
                        <div className="flex justify-between"><span>Bilinear:</span><span>|e| ≲ 10, PSNR ≥ 28 dB</span></div>
                        <div className="flex justify-between"><span>Niu:</span><span>|e| ≲ 45, PSNR ≥ 15 dB*</span></div>
                        <div className="flex justify-between"><span>Hamilton-Adams:</span><span>|e| ≲ 5–10, PSNR ≈ 34–28 dB</span></div>
                        <div className="flex justify-between"><span>Wu:</span><span>|e| ≲ 10, PSNR ≥ 28 dB</span></div>
                        <div className="flex justify-between"><span>Kiku:</span><span>|e| ≲ 2.2, PSNR ≳ 41 dB**</span></div>
                      </div>
                      <p className="text-[10px] text-muted-foreground leading-relaxed mt-2">
                        *Assumes good edge detection. **Empirical bound.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[300px] text-center p-6 border-2 border-dashed border-muted rounded-lg bg-muted/10">
                <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-6 h-6 text-muted-foreground"><path d="M3 3h18v18H3z"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>
                </div>
                <h4 className="font-semibold text-foreground">Error Analysis Requires Ground Truth</h4>
                <p className="text-sm text-muted-foreground mt-1">
                  Load a synthetic or lab image (with known RGB ground truth) to see quantitative error metrics
                  and how they relate to image curvature.
                </p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
