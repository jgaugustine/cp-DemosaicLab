import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Separator } from '@/components/ui/separator';
import { DemosaicCanvas } from '@/components/DemosaicCanvas';
import { DemosaicMathExplanation } from '@/components/DemosaicMathExplanation';
import { BenchmarkMode } from '@/components/BenchmarkMode';
import { DemosaicInput, DemosaicAlgorithm, CFAType, ErrorStats, DemosaicParams } from '@/types/demosaic';
import { simulateCFA } from '@/lib/cfa';
import { 
  demosaicNearest,
  demosaicBilinear,
  demosaicNiuEdgeSensing,
  demosaicLienEdgeBased,
  demosaicWuPolynomial,
  demosaicKikuResidual,
  demosaicXTransNiuEdgeSensing,
  demosaicXTransLienEdgeBased,
  demosaicXTransWuPolynomial,
  demosaicXTransKikuResidual,
  computeErrorStats
} from '@/lib/demosaic';
import { decodeDNG } from '@/lib/dngDecode';
import { createZonePlate, createFineCheckerboard, createColorSweep, createStarburst, createDiagonalLines, createSineWaveGratings, createColorPatches, createColorFringes } from '@/lib/synthetic';
import { Upload, Image as ImageIcon, FileCode, Grid3X3, ZoomIn, ZoomOut, RefreshCcw, Grid, Columns, Loader2 } from 'lucide-react';
import { downsizeImageToDataURL } from '@/lib/imageResize';
import { HelpTooltip } from '@/components/ui/HelpTooltip';

// Viewport configuration type
type ViewportConfig = {
  viewType: 'original' | 'cfa' | 'reconstruction';
  cfaPattern?: CFAType; // For CFA views and reconstruction views (allows per-viewport CFA)
  algorithm?: DemosaicAlgorithm; // For reconstruction views (uses algorithm or algorithm2)
  useAlgorithm2?: boolean; // Whether to use algorithm2 instead of algorithm
};

type ComparisonPreset = 
  | 'original-vs-reconstruction' 
  | 'cfa-vs-reconstruction' 
  | 'algorithm-comparison' 
  | 'cfa-comparison' 
  | 'algorithm-cfa-comparison'
  | '4-up-standard' 
  | 'custom';

export default function Index() {
  const [input, setInput] = useState<DemosaicInput | null>(null);
  const [cfaImage, setCfaImage] = useState<ImageData | null>(null);
  const [cfaImages, setCfaImages] = useState<Record<CFAType, ImageData | null>>({
    bayer: null,
    xtrans: null,
    foveon: null,
  });
  
  // Comparison Mode State
  const [comparisonMode, setComparisonMode] = useState(false);
  const [comparisonPreset, setComparisonPreset] = useState<ComparisonPreset>('algorithm-comparison');
  const [comparisonLayout, setComparisonLayout] = useState<'side-by-side' | '4-up'>('side-by-side');
  const [customViewportConfigs, setCustomViewportConfigs] = useState<ViewportConfig[]>([
    { viewType: 'cfa', cfaPattern: 'bayer' },
    { viewType: 'reconstruction', useAlgorithm2: false },
    { viewType: 'cfa', cfaPattern: 'bayer' },
    { viewType: 'reconstruction', useAlgorithm2: true },
  ]);
  
  // Algo 1 State
  const [outputImage, setOutputImage] = useState<ImageData | null>(null);
  const [algorithm, setAlgorithm] = useState<DemosaicAlgorithm>('bilinear');
  const [errorStats, setErrorStats] = useState<ErrorStats | null>(null);
  const [isProcessing1, setIsProcessing1] = useState(false);
  const [params, setParams] = useState<DemosaicParams>({
    niuLogisticThreshold: 0.1,
    wuPolynomialDegree: 2,
    kikuResidualIterations: 3,
  });

  // Algo 2 State
  const [outputImage2, setOutputImage2] = useState<ImageData | null>(null);
  const [algorithm2, setAlgorithm2] = useState<DemosaicAlgorithm>('nearest');
  const [errorStats2, setErrorStats2] = useState<ErrorStats | null>(null);
  const [isProcessing2, setIsProcessing2] = useState(false);
  const [params2, setParams2] = useState<DemosaicParams>({
    niuLogisticThreshold: 0.1,
    wuPolynomialDegree: 2,
    kikuResidualIterations: 3,
  });

  const [cfaType, setCfaType] = useState<CFAType>('bayer');
  const [uiMode, setUiMode] = useState<'lab' | 'synthetic' | 'raw'>('synthetic');
  const [benchmarkMode, setBenchmarkMode] = useState(false);
  const [syntheticType, setSyntheticType] = useState<string | null>(null);
  const [hoverPos, setHoverPos] = useState<{x: number, y: number} | null>(null);
  const [selectedPos, setSelectedPos] = useState<{x: number, y: number} | null>(null);
  const [viewMode, setViewMode] = useState<'original' | 'cfa' | 'reconstruction'>('reconstruction');
  const [zoom, setZoom] = useState(1);
  const [isFit, setIsFit] = useState(true);
  const isAnyProcessing = isProcessing1 || isProcessing2;
  const [showProcessingOverlay, setShowProcessingOverlay] = useState(false);
  
  // Helper to get algorithm display name (defined early to avoid initialization order issues)
  const getAlgorithmName = (algo: DemosaicAlgorithm): string => {
    switch (algo) {
      case 'nearest':
        return 'Nearest Neighbor';
      case 'bilinear':
        return 'Bilinear Interpolation';
      case 'malvar':
        return 'Malvar';
      case 'high_quality':
        return 'High Quality';
      case 'custom':
        return 'Custom';
      case 'niu_edge_sensing':
        return 'Edge Sensing';
      case 'lien_edge_based':
        return 'Hamilton-Adams (Edge-Based)';
      case 'wu_polynomial':
        return 'Polynomial Interpolation';
      case 'kiku_residual':
        return 'Residual Interpolation';
      default:
        return 'Unknown';
    }
  };
  
  const processingLabel = React.useMemo(() => {
    if (isProcessing1 && isProcessing2) {
      return `Algorithms A (${getAlgorithmName(algorithm)}) & B (${getAlgorithmName(algorithm2)})`;
    }
    if (isProcessing1) {
      return `Algorithm A (${getAlgorithmName(algorithm)})`;
    }
    if (isProcessing2) {
      return `Algorithm B (${getAlgorithmName(algorithm2)})`;
    }
    return '';
  }, [isProcessing1, isProcessing2, algorithm, algorithm2]);

  // Keep the global loading bar visible for at least a short duration
  useEffect(() => {
    if (isAnyProcessing) {
      setShowProcessingOverlay(true);
      return;
    }
    const timeout = window.setTimeout(() => setShowProcessingOverlay(false), 300);
    return () => window.clearTimeout(timeout);
  }, [isAnyProcessing]);
  
  // Cache for reconstruction results
  // Key format: `${inputId}-${algorithm}-${cfaType}-${paramsHash}`
  // Value: { image: ImageData, errorStats: ErrorStats | null }
  const reconstructionCacheRef = useRef<Map<string, { image: ImageData; errorStats: ErrorStats | null }>>(new Map());
  const inputIdRef = useRef<number>(0);
  
  const fileInputRefLab = useRef<HTMLInputElement>(null);
  const fileInputRefRaw = useRef<HTMLInputElement>(null);
  
  const viewport1Ref = useRef<HTMLDivElement>(null);
  const viewport2Ref = useRef<HTMLDivElement>(null);
  const viewport3Ref = useRef<HTMLDivElement>(null);
  const viewport4Ref = useRef<HTMLDivElement>(null);
  const isSyncing = useRef(false);
  
  // Drag-to-pan state
  const [isDragging1, setIsDragging1] = useState(false);
  const [isDragging2, setIsDragging2] = useState(false);
  const [isDragging3, setIsDragging3] = useState(false);
  const [isDragging4, setIsDragging4] = useState(false);
  const dragStartRef1 = useRef<{ x: number; y: number; scrollLeft: number; scrollTop: number } | null>(null);
  const dragStartRef2 = useRef<{ x: number; y: number; scrollLeft: number; scrollTop: number } | null>(null);
  const dragStartRef3 = useRef<{ x: number; y: number; scrollLeft: number; scrollTop: number } | null>(null);
  const dragStartRef4 = useRef<{ x: number; y: number; scrollLeft: number; scrollTop: number } | null>(null);
  const hasDraggedRef1 = useRef(false);
  const hasDraggedRef2 = useRef(false);
  const hasDraggedRef3 = useRef(false);
  const hasDraggedRef4 = useRef(false);
  const DRAG_THRESHOLD = 3; // pixels - only start dragging if mouse moves more than this

  const handleScroll = (source: 1 | 2) => {
    if (isSyncing.current) return;
    const v1 = viewport1Ref.current;
    const v2 = viewport2Ref.current;
    if (!v1 || !v2) return;

    isSyncing.current = true;
    if (source === 1) {
      v2.scrollTop = v1.scrollTop;
      v2.scrollLeft = v1.scrollLeft;
    } else {
      v1.scrollTop = v2.scrollTop;
      v1.scrollLeft = v2.scrollLeft;
    }
    requestAnimationFrame(() => { isSyncing.current = false; });
  };

  // Drag-to-pan handlers for viewport 1
  const handleMouseDown1 = (e: React.MouseEvent<HTMLDivElement>) => {
    if (isFit || !viewport1Ref.current) return;
    // Only start drag on left mouse button
    if (e.button !== 0) return;
    
    hasDraggedRef1.current = false;
    dragStartRef1.current = {
      x: e.clientX,
      y: e.clientY,
      scrollLeft: viewport1Ref.current.scrollLeft,
      scrollTop: viewport1Ref.current.scrollTop,
    };
  };

  const handleMouseMove1 = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!dragStartRef1.current || !viewport1Ref.current || isFit) return;
    
    const deltaX = Math.abs(dragStartRef1.current.x - e.clientX);
    const deltaY = Math.abs(dragStartRef1.current.y - e.clientY);
    
    // Only start dragging if we've moved past the threshold
    if (!isDragging1 && (deltaX > DRAG_THRESHOLD || deltaY > DRAG_THRESHOLD)) {
      setIsDragging1(true);
      hasDraggedRef1.current = true;
      e.preventDefault();
    }
    
    if (!isDragging1 || !dragStartRef1.current) return;
    
    const scrollDeltaX = dragStartRef1.current.x - e.clientX;
    const scrollDeltaY = dragStartRef1.current.y - e.clientY;
    
    viewport1Ref.current.scrollLeft = dragStartRef1.current.scrollLeft + scrollDeltaX;
    viewport1Ref.current.scrollTop = dragStartRef1.current.scrollTop + scrollDeltaY;
    
    // Sync with viewport 2 if in comparison mode
    if (comparisonMode && viewport2Ref.current) {
      viewport2Ref.current.scrollLeft = viewport1Ref.current.scrollLeft;
      viewport2Ref.current.scrollTop = viewport1Ref.current.scrollTop;
    }
    
    e.preventDefault();
  };

  const handleMouseUp1 = (e: React.MouseEvent<HTMLDivElement>) => {
    const wasDragging = hasDraggedRef1.current;
    setIsDragging1(false);
    dragStartRef1.current = null;
    hasDraggedRef1.current = false;
    
    // If we dragged, prevent the click event
    if (wasDragging) {
      e.preventDefault();
      e.stopPropagation();
    }
  };

  const handleClick1 = (e: React.MouseEvent<HTMLDivElement>) => {
    // Prevent clicks on the viewport if we just dragged
    if (hasDraggedRef1.current || isDragging1) {
      e.preventDefault();
      e.stopPropagation();
    }
  };

  // Drag-to-pan handlers for viewport 2
  const handleMouseDown2 = (e: React.MouseEvent<HTMLDivElement>) => {
    if (isFit || !viewport2Ref.current) return;
    if (e.button !== 0) return;
    
    hasDraggedRef2.current = false;
    dragStartRef2.current = {
      x: e.clientX,
      y: e.clientY,
      scrollLeft: viewport2Ref.current.scrollLeft,
      scrollTop: viewport2Ref.current.scrollTop,
    };
  };

  const handleMouseMove2 = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!dragStartRef2.current || !viewport2Ref.current || isFit) return;
    
    const deltaX = Math.abs(dragStartRef2.current.x - e.clientX);
    const deltaY = Math.abs(dragStartRef2.current.y - e.clientY);
    
    // Only start dragging if we've moved past the threshold
    if (!isDragging2 && (deltaX > DRAG_THRESHOLD || deltaY > DRAG_THRESHOLD)) {
      setIsDragging2(true);
      hasDraggedRef2.current = true;
      e.preventDefault();
    }
    
    if (!isDragging2 || !dragStartRef2.current) return;
    
    const scrollDeltaX = dragStartRef2.current.x - e.clientX;
    const scrollDeltaY = dragStartRef2.current.y - e.clientY;
    
    viewport2Ref.current.scrollLeft = dragStartRef2.current.scrollLeft + scrollDeltaX;
    viewport2Ref.current.scrollTop = dragStartRef2.current.scrollTop + scrollDeltaY;
    
    // Sync with viewport 1 if in comparison mode
    if (comparisonMode && viewport1Ref.current) {
      viewport1Ref.current.scrollLeft = viewport2Ref.current.scrollLeft;
      viewport1Ref.current.scrollTop = viewport2Ref.current.scrollTop;
    }
    
    e.preventDefault();
  };

  const handleMouseUp2 = (e: React.MouseEvent<HTMLDivElement>) => {
    const wasDragging = hasDraggedRef2.current;
    setIsDragging2(false);
    dragStartRef2.current = null;
    hasDraggedRef2.current = false;
    
    // If we dragged, prevent the click event
    if (wasDragging) {
      e.preventDefault();
      e.stopPropagation();
    }
  };

  const handleClick2 = (e: React.MouseEvent<HTMLDivElement>) => {
    // Prevent clicks on the viewport if we just dragged
    if (hasDraggedRef2.current || isDragging2) {
      e.preventDefault();
      e.stopPropagation();
    }
  };

  // Global mouse up to handle mouse release outside viewport
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      if (isDragging1) {
        setIsDragging1(false);
        dragStartRef1.current = null;
      }
      if (isDragging2) {
        setIsDragging2(false);
        dragStartRef2.current = null;
      }
      if (isDragging3) {
        setIsDragging3(false);
        dragStartRef3.current = null;
      }
      if (isDragging4) {
        setIsDragging4(false);
        dragStartRef4.current = null;
      }
    };

    if (isDragging1 || isDragging2 || isDragging3 || isDragging4) {
      document.addEventListener('mouseup', handleGlobalMouseUp);
      return () => document.removeEventListener('mouseup', handleGlobalMouseUp);
    }
  }, [isDragging1, isDragging2, isDragging3, isDragging4]);

  const handleZoomIn = () => {
    setIsFit(false);
    setZoom(z => Math.min(z * 1.5, 32));
  };

  const handleZoomOut = () => {
    setIsFit(false);
    setZoom(z => Math.max(z / 1.5, 0.1));
  };

  const toggleFit = () => {
    if (isFit) {
       setIsFit(false);
       setZoom(1);
    } else {
       setIsFit(true);
    }
  };

  const handleWheelZoom = (e: WheelEvent, source: 1 | 2 | 3 | 4) => {
    if (!e.ctrlKey && !e.metaKey) return; // Only zoom with Ctrl/Cmd+Wheel
    e.preventDefault();
    setIsFit(false);
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(z => Math.max(0.1, Math.min(32, z * delta)));
  };

  // Check if original view is available
  const hasGroundTruth = input && (input.mode === 'lab' || input.mode === 'synthetic') && input.groundTruthRGB;

  // Get viewport configurations for a preset
  const getPresetConfigs = useCallback((
    preset: ComparisonPreset,
    layout: 'side-by-side' | '4-up',
    hasGroundTruth: boolean
  ): ViewportConfig[] => {
    if (layout === 'side-by-side') {
      switch (preset) {
        case 'original-vs-reconstruction':
          if (!hasGroundTruth) {
            // Fallback to CFA vs Reconstruction if no ground truth
            return [
              { viewType: 'cfa', cfaPattern: 'bayer' },
              { viewType: 'reconstruction', useAlgorithm2: false },
            ];
          }
          return [
            { viewType: 'original' },
            { viewType: 'reconstruction', useAlgorithm2: false },
          ];
        
        case 'cfa-vs-reconstruction':
          return [
            { viewType: 'cfa', cfaPattern: 'bayer' },
            { viewType: 'reconstruction', useAlgorithm2: false },
          ];
        
        case 'algorithm-comparison':
          return [
            { viewType: 'reconstruction', useAlgorithm2: false },
            { viewType: 'reconstruction', useAlgorithm2: true },
          ];
        
        case 'cfa-comparison':
          return [
            { viewType: 'cfa', cfaPattern: 'bayer' },
            { viewType: 'cfa', cfaPattern: 'xtrans' },
          ];
        
        case 'algorithm-cfa-comparison':
          return [
            { viewType: 'reconstruction', useAlgorithm2: false, cfaPattern: 'bayer' },
            { viewType: 'reconstruction', useAlgorithm2: false, cfaPattern: 'xtrans' },
          ];
        
        default:
          return [
            { viewType: 'cfa', cfaPattern: 'bayer' },
            { viewType: 'reconstruction', useAlgorithm2: false },
          ];
      }
    } else {
      // 4-up layout
      switch (preset) {
        case '4-up-standard':
          return [
            { viewType: 'cfa', cfaPattern: 'bayer' },
            { viewType: 'reconstruction', useAlgorithm2: false },
            { viewType: 'cfa', cfaPattern: 'bayer' },
            { viewType: 'reconstruction', useAlgorithm2: true },
          ];
        
        case 'original-vs-reconstruction':
          if (!hasGroundTruth) {
            return [
              { viewType: 'cfa', cfaPattern: 'bayer' },
              { viewType: 'reconstruction', useAlgorithm2: false },
              { viewType: 'cfa', cfaPattern: 'xtrans' },
              { viewType: 'reconstruction', useAlgorithm2: true },
            ];
          }
          return [
            { viewType: 'original' },
            { viewType: 'reconstruction', useAlgorithm2: false },
            { viewType: 'original' },
            { viewType: 'reconstruction', useAlgorithm2: true },
          ];
        
        case 'cfa-vs-reconstruction':
          return [
            { viewType: 'cfa', cfaPattern: 'bayer' },
            { viewType: 'reconstruction', useAlgorithm2: false },
            { viewType: 'cfa', cfaPattern: 'xtrans' },
            { viewType: 'reconstruction', useAlgorithm2: true },
          ];
        
        case 'algorithm-comparison':
          return [
            { viewType: 'reconstruction', useAlgorithm2: false },
            { viewType: 'reconstruction', useAlgorithm2: false },
            { viewType: 'reconstruction', useAlgorithm2: true },
            { viewType: 'reconstruction', useAlgorithm2: true },
          ];
        
        case 'cfa-comparison':
          return [
            { viewType: 'cfa', cfaPattern: 'bayer' },
            { viewType: 'cfa', cfaPattern: 'bayer' },
            { viewType: 'cfa', cfaPattern: 'xtrans' },
            { viewType: 'cfa', cfaPattern: 'xtrans' },
          ];
        
        case 'algorithm-cfa-comparison':
          return [
            { viewType: 'reconstruction', useAlgorithm2: false, cfaPattern: 'bayer' },
            { viewType: 'reconstruction', useAlgorithm2: false, cfaPattern: 'xtrans' },
            { viewType: 'reconstruction', useAlgorithm2: true, cfaPattern: 'bayer' },
            { viewType: 'reconstruction', useAlgorithm2: true, cfaPattern: 'xtrans' },
          ];
        
        default:
          return [
            { viewType: 'cfa', cfaPattern: 'bayer' },
            { viewType: 'reconstruction', useAlgorithm2: false },
            { viewType: 'cfa', cfaPattern: 'bayer' },
            { viewType: 'reconstruction', useAlgorithm2: true },
          ];
      }
    }
  }, []);

  // Helper to generate label from viewport config
  const getViewportLabel = (config: ViewportConfig): string => {
    switch (config.viewType) {
      case 'original':
        return 'Original';
      case 'cfa':
        if (config.cfaPattern === 'xtrans') return 'X-Trans CFA';
        if (config.cfaPattern === 'foveon') return 'Foveon CFA';
        return 'Bayer CFA';
      case 'reconstruction':
        const algo = config.useAlgorithm2 ? algorithm2 : algorithm;
        const cfa = config.cfaPattern || cfaType;
        const cfaLabel = cfa === 'xtrans' ? 'X-Trans' : cfa === 'foveon' ? 'Foveon' : 'Bayer';
        return `${getAlgorithmName(algo)} (${cfaLabel})`;
      default:
        return 'Unknown';
    }
  };

  // Helper to create a viewport component
  const createViewport = (
    viewportRef: React.RefObject<HTMLDivElement>,
    viewportNum: 1 | 2 | 3 | 4,
    config: ViewportConfig
  ) => {
    const isDragging = viewportNum === 1 ? isDragging1 : viewportNum === 2 ? isDragging2 : viewportNum === 3 ? isDragging3 : isDragging4;
    const dragStartRef = viewportNum === 1 ? dragStartRef1 : viewportNum === 2 ? dragStartRef2 : viewportNum === 3 ? dragStartRef3 : dragStartRef4;
    const hasDraggedRef = viewportNum === 1 ? hasDraggedRef1 : viewportNum === 2 ? hasDraggedRef2 : viewportNum === 3 ? hasDraggedRef3 : hasDraggedRef4;
    
    const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
      if (isFit || !viewportRef.current) return;
      if (e.button !== 0) return;
      
      const target = e.target as HTMLElement;
      if (target.tagName === 'CANVAS') return;
      
      hasDraggedRef.current = false;
      dragStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        scrollLeft: viewportRef.current.scrollLeft,
        scrollTop: viewportRef.current.scrollTop,
      };
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
      if (!dragStartRef.current || !viewportRef.current || isFit) return;
      
      const deltaX = Math.abs(dragStartRef.current.x - e.clientX);
      const deltaY = Math.abs(dragStartRef.current.y - e.clientY);
      
      if (!isDragging && (deltaX > DRAG_THRESHOLD || deltaY > DRAG_THRESHOLD)) {
        if (viewportNum === 1) setIsDragging1(true);
        else if (viewportNum === 2) setIsDragging2(true);
        else if (viewportNum === 3) setIsDragging3(true);
        else setIsDragging4(true);
        hasDraggedRef.current = true;
        e.preventDefault();
      }
      
      if (!isDragging || !dragStartRef.current) return;
      
      const scrollDeltaX = dragStartRef.current.x - e.clientX;
      const scrollDeltaY = dragStartRef.current.y - e.clientY;
      
      viewportRef.current.scrollLeft = dragStartRef.current.scrollLeft + scrollDeltaX;
      viewportRef.current.scrollTop = dragStartRef.current.scrollTop + scrollDeltaY;
      
      // Sync all viewports in comparison mode
      if (comparisonMode && comparisonLayout === '4-up' && viewportRef.current) {
        [viewport1Ref, viewport2Ref, viewport3Ref, viewport4Ref].forEach(ref => {
          if (ref.current && ref !== viewportRef) {
            ref.current.scrollLeft = viewportRef.current.scrollLeft;
            ref.current.scrollTop = viewportRef.current.scrollTop;
          }
        });
      } else if (comparisonMode && comparisonLayout === 'side-by-side' && viewportRef.current) {
        if (viewportNum === 1 && viewport2Ref.current) {
          viewport2Ref.current.scrollLeft = viewportRef.current.scrollLeft;
          viewport2Ref.current.scrollTop = viewportRef.current.scrollTop;
        } else if (viewportNum === 2 && viewport1Ref.current) {
          viewport1Ref.current.scrollLeft = viewportRef.current.scrollLeft;
          viewport1Ref.current.scrollTop = viewportRef.current.scrollTop;
        }
      }
      
      e.preventDefault();
    };

    const handleMouseUp = (e: React.MouseEvent<HTMLDivElement>) => {
      const wasDragging = hasDraggedRef.current;
      if (viewportNum === 1) setIsDragging1(false);
      else if (viewportNum === 2) setIsDragging2(false);
      else if (viewportNum === 3) setIsDragging3(false);
      else setIsDragging4(false);
      dragStartRef.current = null;
      hasDraggedRef.current = false;
      
      if (wasDragging) {
        e.preventDefault();
        e.stopPropagation();
      }
    };

    const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
      if (hasDraggedRef.current || isDragging) {
        e.preventDefault();
        e.stopPropagation();
      }
    };

    const handleScroll = () => {
      if (isSyncing.current || !viewportRef.current) return;
      if (comparisonMode && comparisonLayout === '4-up') {
        [viewport1Ref, viewport2Ref, viewport3Ref, viewport4Ref].forEach(ref => {
          if (ref.current && ref !== viewportRef) {
            isSyncing.current = true;
            ref.current.scrollTop = viewportRef.current.scrollTop;
            ref.current.scrollLeft = viewportRef.current.scrollLeft;
            requestAnimationFrame(() => { isSyncing.current = false; });
          }
        });
      } else if (comparisonMode && comparisonLayout === 'side-by-side') {
        if (viewportNum === 1 && viewport2Ref.current) {
          isSyncing.current = true;
          viewport2Ref.current.scrollTop = viewportRef.current.scrollTop;
          viewport2Ref.current.scrollLeft = viewportRef.current.scrollLeft;
          requestAnimationFrame(() => { isSyncing.current = false; });
        } else if (viewportNum === 2 && viewport1Ref.current) {
          isSyncing.current = true;
          viewport1Ref.current.scrollTop = viewportRef.current.scrollTop;
          viewport1Ref.current.scrollLeft = viewportRef.current.scrollLeft;
          requestAnimationFrame(() => { isSyncing.current = false; });
        }
      }
    };

    const displayImage = getDisplayImage(config);
    const label = getViewportLabel(config);
    // Determine if this viewport should show loading
    const isLoading = config.viewType === 'reconstruction' 
      ? (config.useAlgorithm2 ? isProcessing2 : isProcessing1)
      : false;
    
    // Callback ref to set up native wheel event listener (avoids passive listener issues)
    const setViewportRef = (element: HTMLDivElement | null) => {
      // Update the ref
      if (viewportRef && 'current' in viewportRef) {
        (viewportRef as React.MutableRefObject<HTMLDivElement | null>).current = element;
      }
      
      // Clean up previous listener if element is being removed
      if (!element && viewportRef?.current) {
        const cleanup = (viewportRef.current as any).__wheelCleanup;
        if (cleanup) {
          cleanup();
          delete (viewportRef.current as any).__wheelCleanup;
        }
      }
      
      // Set up wheel listener
      if (element) {
        const wheelHandler = (e: WheelEvent) => handleWheelZoom(e, viewportNum);
        element.addEventListener('wheel', wheelHandler, { passive: false });
        
        // Store cleanup function on the element
        (element as any).__wheelCleanup = () => {
          element.removeEventListener('wheel', wheelHandler);
        };
      }
    };
    
    return (
      <div className="relative h-full w-full min-w-0 overflow-hidden">
        <div 
          ref={setViewportRef}
          onScroll={handleScroll}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onClick={handleClick}
          className={`absolute inset-0 ${!isFit ? 'overflow-auto' : 'overflow-hidden'} bg-[url('/placeholder.svg')] bg-repeat bg-[length:20px_20px] ${!isFit && !isDragging ? 'cursor-grab' : ''} ${!isFit && isDragging ? 'cursor-grabbing' : ''}`}
        >
          <div className="absolute inset-0 opacity-5 pointer-events-none bg-[linear-gradient(45deg,#000_25%,transparent_25%,transparent_75%,#000_75%,#000),linear-gradient(45deg,#000_25%,transparent_25%,transparent_75%,#000_75%,#000)] bg-[length:20px_20px] bg-[position:0_0,10px_10px]"></div>
          <div className="min-w-full min-h-full p-4 flex">
            <div className="m-auto">
              {displayImage && input ? (
                <div className="relative">
                  <DemosaicCanvas 
                    image={displayImage}
                    width={input.width} 
                    height={input.height}
                    className={isFit 
                      ? "max-w-full max-h-full object-contain shadow-2xl border border-border" 
                      : "shadow-2xl border border-border block"}
                    style={!isFit ? { width: input.width * zoom, height: input.height * zoom } : undefined}
                    onPixelHover={(x, y) => setHoverPos(x >= 0 ? {x, y} : null)}
                    onPixelClick={(x, y) => setSelectedPos({x, y})}
                  />
                  <div className="absolute top-2 left-2 bg-background/80 backdrop-blur px-2 py-1 rounded text-[10px] font-mono border border-border z-10 pointer-events-none">
                    {label}
                  </div>
                  {isLoading && (
                    <div className="absolute bottom-2 left-2 bg-background/80 backdrop-blur px-2 py-1 rounded text-[10px] font-mono border border-border z-10 pointer-events-none flex items-center gap-2">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      <span>Processing...</span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center space-y-4 z-10">
                  <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mx-auto border-2 border-dashed border-muted-foreground/50">
                    <ImageIcon className="w-8 h-8 text-muted-foreground" />
                  </div>
                  <div className="text-muted-foreground font-medium">No Image</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  // --- Input Handlers ---

  const loadInputFromImageData = (imageData: ImageData, mode: 'lab' | 'synthetic') => {
    const cfa = simulateCFA(imageData, 'bayer', 'RGGB'); 
    setInput({
      mode: mode,
      groundTruthRGB: imageData,
      cfaPattern: 'bayer',
      cfaPatternMeta: { tileW: 2, tileH: 2, layout: 'RGGB' },
      cfaData: cfa,
      width: imageData.width,
      height: imageData.height
    });
    setCfaType('bayer');
  };

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const dataUrl = await downsizeImageToDataURL(file, 800, 0.9);
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        loadInputFromImageData(imageData, 'lab');
      };
      img.src = dataUrl;
    } catch (err) {
      console.error(err);
    }
  };

  const handleDNGUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const dngInput = await decodeDNG(file);
      setInput(dngInput);
      setCfaType('bayer');
    } catch (err) {
      console.error("DNG Decode failed", err);
      const errorMessage = err instanceof Error ? err.message : "Failed to decode DNG. Ensure it's a valid raw file.";
      alert(errorMessage);
    }
  };

  const handleSyntheticSelect = (type: string) => {
    setSyntheticType(type);
    const w = 512, h = 512;
    let img: ImageData | null = null;
    if (type === 'zoneplate') img = createZonePlate(w, h);
    else if (type === 'checker') img = createFineCheckerboard(w, h, 2);
    else if (type === 'sweep') img = createColorSweep(w, h);
    else if (type === 'star') img = createStarburst(w, h);
    else if (type === 'diagonal') img = createDiagonalLines(w, h);
    else if (type === 'sine') img = createSineWaveGratings(w, h);
    else if (type === 'patches') img = createColorPatches(w, h);
    else if (type === 'fringes') img = createColorFringes(w, h);
    if (img) loadInputFromImageData(img, 'synthetic');
  };
  
  // Sync input CFA when user changes CFA Type
  useEffect(() => {
    if (!input || input.mode === 'raw' || !input.groundTruthRGB) return;
    if (input.cfaPattern === cfaType) return;
    const newCfa = simulateCFA(input.groundTruthRGB, cfaType);
    setInput(prev => prev ? ({
      ...prev,
      cfaPattern: cfaType,
      cfaPatternMeta: {
        tileW: cfaType === 'xtrans' ? 6 : 2,
        tileH: cfaType === 'xtrans' ? 6 : 2,
        layout: cfaType === 'bayer' ? 'RGGB' : 'custom'
      },
      cfaData: newCfa
    }) : null);
  }, [cfaType, input]);

  // Helper to generate CFA visualization for a given pattern
  const generateCFAVisualization = useCallback((cfaData: Float32Array | Uint16Array, pattern: CFAType, width: number, height: number): ImageData => {
    const img = new ImageData(width, height);
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const v = cfaData[y * width + x] * 255;
        const idx = (y * width + x) * 4;
            
            // Determine color
            let isR = false, isG = false, isB = false;
            
        if (pattern === 'bayer') {
                // RGGB
                const isEvenY = y % 2 === 0;
                const isEvenX = x % 2 === 0;
                if (isEvenY) {
                    if (isEvenX) isR = true; else isG = true;
                } else if (isEvenX) {
                    isG = true;
                } else {
                    isB = true;
                }
        } else if (pattern === 'xtrans') {
                // 6x6 Pattern
          const patternLayout = [
                    ['g', 'r', 'g', 'g', 'b', 'g'],
                    ['b', 'g', 'b', 'r', 'g', 'r'],
                    ['g', 'r', 'g', 'g', 'b', 'g'],
                    ['g', 'b', 'g', 'g', 'r', 'g'],
                    ['r', 'g', 'r', 'b', 'g', 'b'],
                    ['g', 'b', 'g', 'g', 'r', 'g'],
                ];
          const ch = patternLayout[y % 6][x % 6];
                if (ch === 'r') isR = true;
                else if (ch === 'g') isG = true;
                else isB = true;
            }
            
            img.data[idx] = isR ? v : 0;
            img.data[idx+1] = isG ? v : 0;
            img.data[idx+2] = isB ? v : 0;
            img.data[idx+3] = 255;
        }
    }
    
    return img;
  }, []);

  // Update input ID when input changes (to invalidate cache)
  // We use a ref to track the previous input to detect actual changes
  const prevInputRef = useRef<DemosaicInput | null>(null);
  useEffect(() => {
    if (!input) {
      prevInputRef.current = null;
      return;
    }
    
    const prevInput = prevInputRef.current;
    // Check if input actually changed by comparing key properties
    const inputChanged = !prevInput || 
      prevInput.width !== input.width ||
      prevInput.height !== input.height ||
      prevInput.mode !== input.mode ||
      prevInput.cfaPattern !== input.cfaPattern ||
      prevInput.cfaData !== input.cfaData; // Reference comparison for CFA data
    
    if (inputChanged) {
      inputIdRef.current += 1;
      // Clear cache when input changes
      reconstructionCacheRef.current.clear();
      prevInputRef.current = input;
    }
  }, [input]);

  // Generate CFA Visualization Images for all patterns
  useEffect(() => {
    if (!input) {
      setCfaImage(null);
      setCfaImages({ bayer: null, xtrans: null, foveon: null });
      return;
    }
    
    // Check if we already have CFA images cached for this input
    // We'll regenerate only if the input actually changed (handled by the dependency on input)

    // For raw mode, we only have the input's CFA pattern
    if (input.mode === 'raw') {
      const img = generateCFAVisualization(input.cfaData, input.cfaPattern, input.width, input.height);
    setCfaImage(img);
      const newCfaImages: Record<CFAType, ImageData | null> = {
        bayer: input.cfaPattern === 'bayer' ? img : null,
        xtrans: input.cfaPattern === 'xtrans' ? img : null,
        foveon: input.cfaPattern === 'foveon' ? img : null,
      };
      setCfaImages(newCfaImages);
      return;
    }

    // For lab/synthetic mode, we can generate CFAs for all patterns from ground truth
    if (input.groundTruthRGB) {
      const newCfaImages: Record<CFAType, ImageData | null> = {
        bayer: null,
        xtrans: null,
        foveon: null,
      };

      // Generate Bayer CFA
      const bayerCfa = simulateCFA(input.groundTruthRGB, 'bayer');
      newCfaImages.bayer = generateCFAVisualization(bayerCfa, 'bayer', input.width, input.height);

      // Generate X-Trans CFA
      const xtransCfa = simulateCFA(input.groundTruthRGB, 'xtrans');
      newCfaImages.xtrans = generateCFAVisualization(xtransCfa, 'xtrans', input.width, input.height);

      // Set the current CFA image based on input's pattern
      setCfaImage(newCfaImages[input.cfaPattern] || newCfaImages.bayer);
      setCfaImages(newCfaImages);
    } else {
      // Fallback: use input's CFA
      const img = generateCFAVisualization(input.cfaData, input.cfaPattern, input.width, input.height);
      setCfaImage(img);
      const newCfaImages: Record<CFAType, ImageData | null> = {
        bayer: input.cfaPattern === 'bayer' ? img : null,
        xtrans: input.cfaPattern === 'xtrans' ? img : null,
        foveon: input.cfaPattern === 'foveon' ? img : null,
      };
      setCfaImages(newCfaImages);
    }
  }, [input, generateCFAVisualization]);

  // Helper to create a cache key for reconstruction results
  const createCacheKey = useCallback((
    inputId: number,
    algo: DemosaicAlgorithm,
    cfa: CFAType,
    algoParams?: DemosaicParams
  ): string => {
    const paramsStr = algoParams 
      ? `${algoParams.niuLogisticThreshold}-${algoParams.wuPolynomialDegree}`
      : 'default';
    return `${inputId}-${algo}-${cfa}-${paramsStr}`;
  }, []);

  // Helper to run demosaic with caching
  const runDemosaic = useCallback((inp: DemosaicInput, algo: DemosaicAlgorithm, algoParams?: DemosaicParams) => {
    // Use X-Trans specific implementations when CFA pattern is X-Trans
    if (inp.cfaPattern === 'xtrans') {
      if (algo === 'nearest') return demosaicNearest(inp);
      if (algo === 'bilinear') return demosaicBilinear(inp);
      if (algo === 'niu_edge_sensing') return demosaicXTransNiuEdgeSensing(inp, algoParams);
      if (algo === 'lien_edge_based') return demosaicXTransLienEdgeBased(inp);
      if (algo === 'wu_polynomial') return demosaicXTransWuPolynomial(inp, algoParams);
      if (algo === 'kiku_residual') return demosaicXTransKikuResidual(inp, algoParams);
      return new ImageData(inp.width, inp.height);
    }
    
    // Use Bayer/generic implementations for Bayer pattern
    if (algo === 'nearest') return demosaicNearest(inp);
    if (algo === 'bilinear') return demosaicBilinear(inp);
    if (algo === 'niu_edge_sensing') return demosaicNiuEdgeSensing(inp, algoParams);
    if (algo === 'lien_edge_based') return demosaicLienEdgeBased(inp);
    if (algo === 'wu_polynomial') return demosaicWuPolynomial(inp, algoParams);
    if (algo === 'kiku_residual') return demosaicKikuResidual(inp, algoParams);
    return new ImageData(inp.width, inp.height);
  }, []);

  // Helper to get or compute reconstruction result with caching
  const getOrComputeReconstruction = useCallback((
    inp: DemosaicInput,
    algo: DemosaicAlgorithm,
    cfa: CFAType,
    algoParams?: DemosaicParams
  ): { image: ImageData; errorStats: ErrorStats | null } => {
    const cacheKey = createCacheKey(inputIdRef.current, algo, cfa, algoParams);
    const cached = reconstructionCacheRef.current.get(cacheKey);
    
    if (cached) {
      return cached;
    }
    
    // If the requested CFA is different from the input's CFA, create a modified input
    let demosaicInput = inp;
    if (cfa !== inp.cfaPattern) {
      // Generate new CFA data for the requested pattern
      let newCfaData: Float32Array | Uint16Array;
      
      if ((inp.mode === 'lab' || inp.mode === 'synthetic') && inp.groundTruthRGB) {
        // Generate CFA from ground truth
        newCfaData = simulateCFA(inp.groundTruthRGB, cfa);
      } else {
        // For raw mode, we can't generate a different CFA, so use the original
        // This shouldn't happen in practice for algorithm-cfa-comparison mode
        newCfaData = inp.cfaData;
      }
      
      // Create modified input with the new CFA pattern
      demosaicInput = {
        ...inp,
        cfaPattern: cfa,
        cfaPatternMeta: {
          tileW: cfa === 'xtrans' ? 6 : 2,
          tileH: cfa === 'xtrans' ? 6 : 2,
          layout: cfa === 'bayer' ? 'RGGB' : 'custom'
        },
        cfaData: newCfaData
      };
    }
    
    // Compute new result with the (possibly modified) input
    const image = runDemosaic(demosaicInput, algo, algoParams);
    let errorStats: ErrorStats | null = null;
    if ((inp.mode === 'lab' || inp.mode === 'synthetic') && inp.groundTruthRGB) {
      errorStats = computeErrorStats(inp.groundTruthRGB, image);
    }
    
    const result = { image, errorStats };
    reconstructionCacheRef.current.set(cacheKey, result);
    return result;
  }, [createCacheKey, runDemosaic]);

  // Helper to get reconstruction for a specific algorithm+CFA combination
  const getReconstructionForConfig = useCallback((config: ViewportConfig): ImageData | null => {
    if (!input || config.viewType !== 'reconstruction') return null;
    
    const algo = config.useAlgorithm2 ? algorithm2 : algorithm;
    const algoParams = config.useAlgorithm2 ? params2 : params;
    const cfa = config.cfaPattern || cfaType; // Use viewport-specific CFA or fallback to global
    
    // Check cache first
    const cacheKey = createCacheKey(inputIdRef.current, algo, cfa, algoParams);
    const cached = reconstructionCacheRef.current.get(cacheKey);
    
    if (cached) {
      return cached.image;
    }
    
    // If not cached, compute it synchronously (this might be slow, but needed for viewport display)
    // In production, you might want to pre-compute these
    try {
      const result = getOrComputeReconstruction(input, algo, cfa, algoParams);
      return result.image;
    } catch {
      return null;
    }
  }, [input, algorithm, algorithm2, params, params2, cfaType, createCacheKey, getOrComputeReconstruction]);

  // Determine which image to display based on viewport config
  const getDisplayImage = useCallback((config: ViewportConfig): ImageData | null => {
    if (!input) return null;
    
    switch (config.viewType) {
      case 'original':
        return input.groundTruthRGB || null;
      
      case 'cfa':
        if (config.cfaPattern) {
          return cfaImages[config.cfaPattern] || null;
        }
        // Fallback to current CFA image if pattern not specified
        return cfaImage;
      
      case 'reconstruction':
        // If viewport has a specific CFA pattern, use that; otherwise use default behavior
        if (config.cfaPattern) {
          return getReconstructionForConfig(config);
        }
        // Fallback to standard output images
        return config.useAlgorithm2 ? outputImage2 : outputImage;
      
      default:
        return outputImage;
    }
  }, [input, cfaImage, cfaImages, outputImage, outputImage2, getReconstructionForConfig]);

  // Pipeline 1
  useEffect(() => {
    if (!input) return;
    
    // Check cache first
    const cacheKey = createCacheKey(inputIdRef.current, algorithm, cfaType, params);
    const cached = reconstructionCacheRef.current.get(cacheKey);
    
    if (cached) {
      // Use cached result
      setOutputImage(cached.image);
      setErrorStats(cached.errorStats);
      setIsProcessing1(false);
      return;
    }
    
    // Not in cache, compute it
    setIsProcessing1(true);
    // Use requestAnimationFrame to allow UI to update before heavy computation
    requestAnimationFrame(() => {
      // Use another frame to ensure the loading state renders
      requestAnimationFrame(() => {
        const result = getOrComputeReconstruction(input, algorithm, cfaType, params);
        setOutputImage(result.image);
        setErrorStats(result.errorStats);
        setIsProcessing1(false);
      });
    });
  }, [input, algorithm, cfaType, params, createCacheKey, getOrComputeReconstruction]);

  // Pipeline 2 (Comparison)
  useEffect(() => {
    if (!input || !comparisonMode) {
        setOutputImage2(null);
        setIsProcessing2(false);
        return;
    }
    
    // Check cache first
    const cacheKey = createCacheKey(inputIdRef.current, algorithm2, cfaType, params2);
    const cached = reconstructionCacheRef.current.get(cacheKey);
    
    if (cached) {
      // Use cached result
      setOutputImage2(cached.image);
      setErrorStats2(cached.errorStats);
      setIsProcessing2(false);
      return;
    }
    
    // Not in cache, compute it
    setIsProcessing2(true);
    // Use requestAnimationFrame to allow UI to update before heavy computation
    requestAnimationFrame(() => {
      // Use another frame to ensure the loading state renders
      requestAnimationFrame(() => {
        const result = getOrComputeReconstruction(input, algorithm2, cfaType, params2);
        setOutputImage2(result.image);
        setErrorStats2(result.errorStats);
        setIsProcessing2(false);
      });
    });
  }, [input, algorithm2, cfaType, comparisonMode, params2, createCacheKey, getOrComputeReconstruction]);

  // Initialize viewport configs when comparison mode or layout changes
  useEffect(() => {
    if (!comparisonMode) return;
    
    if (comparisonPreset !== 'custom') {
      const configs = getPresetConfigs(comparisonPreset, comparisonLayout, !!hasGroundTruth);
      // Always keep 4 configs for 4-up, but only use first 2 for side-by-side
      if (comparisonLayout === 'side-by-side') {
        const sideBySideConfigs = configs.slice(0, 2);
        // Pad with defaults if needed
        const padded = [
          ...sideBySideConfigs,
          { viewType: 'cfa' as const, cfaPattern: 'bayer' as const },
          { viewType: 'cfa' as const, cfaPattern: 'bayer' as const },
        ];
        setCustomViewportConfigs(padded);
      } else {
        // Ensure we have 4 configs for 4-up
        const fourUpConfigs = configs.slice(0, 4);
        const padded = [
          ...fourUpConfigs,
          ...Array(4).fill({ viewType: 'cfa' as const, cfaPattern: 'bayer' as const }).slice(0, 4 - fourUpConfigs.length)
        ];
        setCustomViewportConfigs(padded);
      }
    }
  }, [comparisonMode, comparisonPreset, comparisonLayout, hasGroundTruth, getPresetConfigs]);

  // Reset view mode when input changes or when ground truth becomes unavailable
  useEffect(() => {
    if (!input || (input.mode !== 'lab' && input.mode !== 'synthetic')) {
      // For raw mode or no input, default to reconstruction or CFA
      if (viewMode === 'original') setViewMode('reconstruction');
    }
  }, [input, viewMode]);

  // Render benchmark mode if enabled
  if (benchmarkMode) {
    return (
      <div className="h-screen bg-background text-foreground flex flex-col overflow-hidden">
        <div className="border-b p-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold">Benchmark Mode</h1>
          <Button variant="outline" onClick={() => setBenchmarkMode(false)}>
            Exit Benchmark Mode
          </Button>
        </div>
        <div className="flex-1 overflow-auto">
          <BenchmarkMode 
            defaultParams={params}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-background text-foreground flex flex-col overflow-hidden">
      <main className="flex-1 p-4 lg:p-6 min-h-0 pt-2">
        <div className="w-full h-full max-w-[1600px] mx-auto grid lg:grid-cols-12 gap-6">
          
          {/* Left Panel: Controls */}
          <div className="lg:col-span-3 flex flex-col gap-4 h-full overflow-y-auto pr-2 min-w-0">
            {/* 1. Mode Selection */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-3">
                <h2 className="text-lg font-semibold text-primary flex items-center gap-2">
                  Mode
                  <HelpTooltip className="h-3 w-3" content="
                    Synthetic: Mathematical patterns to test algorithms.
                    JPEG Lab: Standard images (JPG/PNG) treated as ground truth to simulate sensor sampling.
                    Real Raw: Actual sensor data from DNG files.
                  " />
                </h2>
              </CardHeader>
              <CardContent>
                <ToggleGroup 
                  type="single" 
                  value={uiMode} 
                  onValueChange={(v) => {
                    if (v && (v === 'synthetic' || v === 'raw' || v === 'lab')) {
                      setUiMode(v);
                    }
                  }}
                  variant="outline"
                  className="w-full"
                >
                  <ToggleGroupItem value="synthetic" aria-label="Synthetic mode" className="flex-1 text-xs bg-background">
                    Synthetic
                  </ToggleGroupItem>
                  <ToggleGroupItem value="lab" aria-label="JPEG Lab mode" className="flex-1 text-xs bg-background">
                    JPEG Lab
                  </ToggleGroupItem>
                  <ToggleGroupItem value="raw" aria-label="Real Raw mode" className="flex-1 text-xs bg-background">
                    Real Raw
                  </ToggleGroupItem>
                </ToggleGroup>
                <Button 
                  variant="outline" 
                  className="w-full mt-3 !text-xs"
                  onClick={() => setBenchmarkMode(true)}
                >
                  Benchmark Mode
                </Button>
              </CardContent>
            </Card>

            {/* 2. Select Image */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-3">
                <h2 className="text-lg font-semibold text-primary">Select Image</h2>
              </CardHeader>
              <CardContent className="space-y-4">

                {uiMode === 'synthetic' && (
                  <div className="space-y-2">
                    <label className="text-xs font-medium mb-2 text-muted-foreground flex items-center gap-2">
                      Synthetic Patterns
                      <HelpTooltip className="h-3 w-3" content="Procedurally generated patterns designed to stress-test demosaicing algorithms. Zone plates show Moiré, Starbursts show edge handling." />
                    </label>
                    <Select value={syntheticType || ""} onValueChange={handleSyntheticSelect}>
                      <SelectTrigger><SelectValue placeholder="Select Pattern..." /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="zoneplate">Zone Plate (Moiré)</SelectItem>
                        <SelectItem value="checker">Fine Checkerboard</SelectItem>
                        <SelectItem value="sweep">Color Sweep</SelectItem>
                        <SelectItem value="star">Starburst</SelectItem>
                        <SelectItem value="diagonal">Diagonal Lines</SelectItem>
                        <SelectItem value="sine">Sine Wave Gratings</SelectItem>
                        <SelectItem value="patches">Color Patches</SelectItem>
                        <SelectItem value="fringes">Color Fringes</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                )}

                {uiMode === 'lab' && (
                  <div className="space-y-2">
                    <Button variant="outline" className="w-full h-24 flex flex-col gap-2 border-dashed" onClick={() => fileInputRefLab.current?.click()}>
                      <ImageIcon className="w-8 h-8 mb-1 text-muted-foreground" />
                      <div className="text-center">
                        <span className="text-xs block font-medium">Upload JPEG Lab Image</span>
                        <span className="text-[10px] text-muted-foreground">(JPEG, PNG)</span>
                      </div>
                    </Button>
                    <input ref={fileInputRefLab} type="file" accept="image/*" className="hidden" onChange={handleImageUpload} />
                    <p className="text-[10px] text-muted-foreground text-center leading-tight px-1">
                      Images are converted to a simulated raw sensor mosaic (CFA) and then reconstructed. This allows comparing algorithms against the original ground truth.
                    </p>
                  </div>
                )}

                {uiMode === 'raw' && (
                  <div className="space-y-2">
                    <Button variant="outline" className="w-full h-24 flex flex-col gap-2 border-dashed" onClick={() => fileInputRefRaw.current?.click()}>
                      <FileCode className="w-8 h-8 mb-1 text-muted-foreground" />
                      <div className="text-center">
                        <span className="text-xs block font-medium">Upload RAW File</span>
                        <span className="text-[10px] text-muted-foreground">(DNG)</span>
                      </div>
                    </Button>
                    <input ref={fileInputRefRaw} type="file" accept=".dng,.tif" className="hidden" onChange={handleDNGUpload} />
                  </div>
                )}

                {input && (
                  <div className="bg-muted/50 rounded p-2 text-xs font-mono flex justify-between mt-2">
                    <span>{input.width} x {input.height}</span><span className="uppercase">{input.mode} Mode</span>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 3. Choose CFA */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-3">
                <h2 className="text-lg font-semibold text-primary flex items-center gap-2">
                  Choose CFA
                  <Grid3X3 className="w-3 h-3" />
                </h2>
              </CardHeader>
              <CardContent>
                <ToggleGroup 
                  type="single" 
                  value={cfaType} 
                  onValueChange={(v) => {
                    if (v && (v === 'bayer' || v === 'xtrans')) {
                      setCfaType(v as CFAType);
                    }
                  }}
                  disabled={input?.mode === 'raw'}
                  className="w-full"
                >
                  <ToggleGroupItem 
                    value="bayer" 
                    aria-label="Bayer CFA" 
                    className="flex-1 text-xs bg-background border border-input hover:bg-accent/60 hover:text-accent-foreground data-[state=on]:bg-accent data-[state=on]:text-accent-foreground" 
                    disabled={input?.mode === 'raw'}
                  >
                    Bayer (RGGB)
                  </ToggleGroupItem>
                  <ToggleGroupItem 
                    value="xtrans" 
                    aria-label="X-Trans CFA" 
                    className="flex-1 text-xs bg-background border border-input hover:bg-accent/60 hover:text-accent-foreground data-[state=on]:bg-accent data-[state=on]:text-accent-foreground" 
                    disabled={input?.mode === 'raw'}
                  >
                    X-Trans (6x6)
                  </ToggleGroupItem>
                </ToggleGroup>
              </CardContent>
            </Card>

            {/* 4. Choose Algorithm */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-3">
                <h2 className="text-lg font-semibold text-primary">Choose Algorithm</h2>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-xs font-medium text-primary">Algorithm A</label>
                  <Select value={algorithm} onValueChange={(v) => setAlgorithm(v as DemosaicAlgorithm)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="nearest">Nearest Neighbor</SelectItem>
                      <SelectItem value="bilinear">Bilinear Interpolation</SelectItem>
                      <SelectItem value="niu_edge_sensing">Edge Sensing</SelectItem>
                      <SelectItem value="wu_polynomial">Polynomial Interpolation</SelectItem>
                      <SelectItem value="lien_edge_based">Hamilton-Adams (Edge-Based)</SelectItem>
                      <SelectItem value="kiku_residual">Residual Interpolation</SelectItem>
                    </SelectContent>
                  </Select>
                  {errorStats && (
                    <div className="flex items-center justify-between text-[10px] text-muted-foreground px-1">
                      <div className="flex items-center gap-1">
                        <span>PSNR: {errorStats.psnr.total.toFixed(2)} dB</span>
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
                      <div className="flex items-center gap-1">
                        <span>MSE: {errorStats.mse.total.toFixed(2)}</span>
                        <HelpTooltip className="h-3 w-3" content="Mean Squared Error. Lower is better. Average squared difference between estimated and true pixel values." />
                      </div>
                    </div>
                  )}
                </div>

                {comparisonMode && (
                  <div className="space-y-2 pt-2 border-t animate-in fade-in slide-in-from-top-2">
                    <label className="text-xs font-medium text-primary">Algorithm B</label>
                    <Select value={algorithm2} onValueChange={(v) => setAlgorithm2(v as DemosaicAlgorithm)}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="nearest">Nearest Neighbor</SelectItem>
                        <SelectItem value="bilinear">Bilinear Interpolation</SelectItem>
                        <SelectItem value="niu_edge_sensing">Edge Sensing</SelectItem>
                        <SelectItem value="wu_polynomial">Polynomial Interpolation</SelectItem>
                        <SelectItem value="lien_edge_based">Hamilton-Adams (Edge-Based)</SelectItem>
                        <SelectItem value="kiku_residual">Residual Interpolation</SelectItem>
                      </SelectContent>
                    </Select>
                    {errorStats2 && (
                      <div className="flex items-center justify-between text-[10px] text-muted-foreground px-1">
                        <div className="flex items-center gap-1">
                          <span>PSNR: {errorStats2.psnr.total.toFixed(2)} dB</span>
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
                        <div className="flex items-center gap-1">
                          <span>MSE: {errorStats2.mse.total.toFixed(2)}</span>
                          <HelpTooltip className="h-3 w-3" content="Mean Squared Error. Lower is better. Average squared difference between estimated and true pixel values." />
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 5. Hyperparameters */}
            {((algorithm === 'niu_edge_sensing' || algorithm === 'wu_polynomial' || algorithm === 'kiku_residual') || 
              (comparisonMode && (algorithm2 === 'niu_edge_sensing' || algorithm2 === 'wu_polynomial' || algorithm2 === 'kiku_residual'))) && (
              <Card className="bg-card border-border">
                <CardHeader className="pb-3">
                  <h2 className="text-lg font-semibold text-primary">Hyperparameters</h2>
                </CardHeader>
                <CardContent className="space-y-4">
                  {(algorithm === 'niu_edge_sensing' || algorithm === 'wu_polynomial' || algorithm === 'kiku_residual') && (
                    <div className="space-y-3">
                      <label className="text-xs font-medium text-muted-foreground">Algorithm A</label>
                      {algorithm === 'niu_edge_sensing' && (
                        <div className="space-y-4">
                          <div className="space-y-2">
                            <div className="flex items-center justify-between text-[10px]">
                              <span className="text-muted-foreground">Edge Threshold</span>
                              <span className="font-mono">{params.niuLogisticThreshold?.toFixed(3) ?? '0.100'}</span>
                            </div>
                            <div
                              onDoubleClick={() => setParams(prev => {
                                const { niuLogisticThreshold, ...rest } = prev;
                                return rest;
                              })}
                            >
                              <Slider
                                value={[params.niuLogisticThreshold ?? 0.1]}
                                onValueChange={([v]) => setParams(prev => ({ ...prev, niuLogisticThreshold: v }))}
                                min={0.001}
                                max={1.0}
                                step={0.001}
                                className="w-full"
                              />
                            </div>
                            <HelpTooltip className="h-3 w-3" content="Threshold for edge detection in the logistic function. Lower values detect more edges, higher values are more selective. Double-click to reset to default (0.1)." />
                          </div>
                          <div className="space-y-2">
                            <div className="flex items-center justify-between text-[10px]">
                              <span className="text-muted-foreground">Logistic Steepness</span>
                              <span className="font-mono">{params.niuLogisticSteepness?.toFixed(1) ?? '20.0'}</span>
                            </div>
                            <div
                              onDoubleClick={() => setParams(prev => {
                                const { niuLogisticSteepness, ...rest } = prev;
                                return rest;
                              })}
                            >
                              <Slider
                                value={[params.niuLogisticSteepness ?? 20.0]}
                                onValueChange={([v]) => setParams(prev => ({ ...prev, niuLogisticSteepness: v }))}
                                min={1.0}
                                max={200.0}
                                step={0.1}
                                className="w-full"
                              />
                            </div>
                            <HelpTooltip className="h-3 w-3" content="Steepness parameter (k) for the logistic function. Higher values create a sharper transition, lower values create a more gradual transition. Double-click to reset to default (adaptive)." />
                          </div>
                        </div>
                      )}
                      {algorithm === 'wu_polynomial' && (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-[10px]">
                            <span className="text-muted-foreground">Polynomial Degree</span>
                            <span className="font-mono">{params.wuPolynomialDegree ?? 2}</span>
                          </div>
                          <div
                            onDoubleClick={() => setParams(prev => {
                              const { wuPolynomialDegree, ...rest } = prev;
                              return rest;
                            })}
                          >
                            <Slider
                              value={[params.wuPolynomialDegree ?? 2]}
                              onValueChange={([v]) => setParams(prev => ({ ...prev, wuPolynomialDegree: Math.round(v) }))}
                              min={1}
                              max={10}
                              step={1}
                              className="w-full"
                            />
                          </div>
                          <HelpTooltip className="h-3 w-3" content="Degree of polynomial interpolation. Higher degrees can capture more complex patterns but may overfit. Double-click to reset to default (2)." />
                        </div>
                      )}
                      {algorithm === 'kiku_residual' && (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-[10px]">
                            <span className="text-muted-foreground">Residual Iterations</span>
                            <span className="font-mono">{params.kikuResidualIterations ?? 1}</span>
                          </div>
                          <div
                            onDoubleClick={() => setParams(prev => {
                              const { kikuResidualIterations, ...rest } = prev;
                              return rest;
                            })}
                          >
                            <Slider
                              value={[params.kikuResidualIterations ?? 1]}
                              onValueChange={([v]) => setParams({ ...params, kikuResidualIterations: Math.round(v) })}
                              min={1}
                              max={5}
                              step={1}
                              className="w-full"
                            />
                          </div>
                          <HelpTooltip className="h-3 w-3" content="Number of residual refinement iterations. More iterations can improve quality but increase computation time. Double-click to reset to default (1)." />
                        </div>
                      )}
                    </div>
                  )}

                  {comparisonMode && (algorithm2 === 'niu_edge_sensing' || algorithm2 === 'wu_polynomial' || algorithm2 === 'kiku_residual') && (
                    <div className="space-y-3 pt-3 border-t animate-in fade-in slide-in-from-top-2">
                      <label className="text-xs font-medium text-muted-foreground">Algorithm B</label>
                      {algorithm2 === 'niu_edge_sensing' && (
                        <div className="space-y-4">
                          <div className="space-y-2">
                            <div className="flex items-center justify-between text-[10px]">
                              <span className="text-muted-foreground">Edge Threshold</span>
                              <span className="font-mono">{params2.niuLogisticThreshold?.toFixed(3) ?? '0.100'}</span>
                            </div>
                            <div
                              onDoubleClick={() => setParams2(prev => {
                                const { niuLogisticThreshold, ...rest } = prev;
                                return rest;
                              })}
                            >
                              <Slider
                                value={[params2.niuLogisticThreshold ?? 0.1]}
                                onValueChange={([v]) => setParams2(prev => ({ ...prev, niuLogisticThreshold: v }))}
                                min={0.001}
                                max={1.0}
                                step={0.001}
                                className="w-full"
                              />
                            </div>
                            <HelpTooltip className="h-3 w-3" content="Threshold for edge detection in the logistic function. Lower values detect more edges, higher values are more selective. Double-click to reset to default (0.1)." />
                          </div>
                          <div className="space-y-2">
                            <div className="flex items-center justify-between text-[10px]">
                              <span className="text-muted-foreground">Logistic Steepness</span>
                              <span className="font-mono">{params2.niuLogisticSteepness?.toFixed(1) ?? '20.0'}</span>
                            </div>
                            <div
                              onDoubleClick={() => setParams2(prev => {
                                const { niuLogisticSteepness, ...rest } = prev;
                                return rest;
                              })}
                            >
                              <Slider
                                value={[params2.niuLogisticSteepness ?? 20.0]}
                                onValueChange={([v]) => setParams2(prev => ({ ...prev, niuLogisticSteepness: v }))}
                                min={1.0}
                                max={200.0}
                                step={0.1}
                                className="w-full"
                              />
                            </div>
                            <HelpTooltip className="h-3 w-3" content="Steepness parameter (k) for the logistic function. Higher values create a sharper transition, lower values create a more gradual transition. Double-click to reset to default (adaptive)." />
                          </div>
                        </div>
                      )}
                      {algorithm2 === 'wu_polynomial' && (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-[10px]">
                            <span className="text-muted-foreground">Polynomial Degree</span>
                            <span className="font-mono">{params2.wuPolynomialDegree ?? 2}</span>
                          </div>
                          <div
                            onDoubleClick={() => setParams2(prev => {
                              const { wuPolynomialDegree, ...rest } = prev;
                              return rest;
                            })}
                          >
                            <Slider
                              value={[params2.wuPolynomialDegree ?? 2]}
                              onValueChange={([v]) => setParams2(prev => ({ ...prev, wuPolynomialDegree: Math.round(v) }))}
                              min={1}
                              max={10}
                              step={1}
                              className="w-full"
                            />
                          </div>
                          <HelpTooltip className="h-3 w-3" content="Degree of polynomial interpolation. Higher degrees can capture more complex patterns but may overfit. Double-click to reset to default (2)." />
                        </div>
                      )}
                      {algorithm2 === 'kiku_residual' && (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-[10px]">
                            <span className="text-muted-foreground">Residual Iterations</span>
                            <span className="font-mono">{params2.kikuResidualIterations ?? 1}</span>
                          </div>
                          <div
                            onDoubleClick={() => setParams2(prev => {
                              const { kikuResidualIterations, ...rest } = prev;
                              return rest;
                            })}
                          >
                            <Slider
                              value={[params2.kikuResidualIterations ?? 1]}
                              onValueChange={([v]) => setParams2({ ...params2, kikuResidualIterations: Math.round(v) })}
                              min={1}
                              max={5}
                              step={1}
                              className="w-full"
                            />
                          </div>
                          <HelpTooltip className="h-3 w-3" content="Number of residual refinement iterations. More iterations can improve quality but increase computation time. Double-click to reset to default (1)." />
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* 6. View Setup */}
            <Card className="bg-card border-border">
              <CardHeader className="pb-3">
                <h2 className="text-lg font-semibold text-primary">View Setup</h2>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-xs font-medium flex items-center gap-2">
                    View Mode
                    <HelpTooltip className="h-3 w-3" content={
                      hasGroundTruth 
                        ? "Original: The ground truth image before CFA sampling. CFA: What the sensor sees (one channel per pixel). Reconstruction: The demosaiced result."
                        : "CFA: What the sensor sees (one channel per pixel). Reconstruction: The demosaiced result."
                    } />
                  </label>
                  <ToggleGroup 
                    type="single" 
                    value={viewMode} 
                    onValueChange={(v) => {
                      if (v && (v === 'original' || v === 'cfa' || v === 'reconstruction')) {
                        setViewMode(v);
                      }
                    }}
                    className="w-full"
                  >
                    {hasGroundTruth && (
                      <ToggleGroupItem 
                        value="original" 
                        aria-label="Show original" 
                        className="flex-1 text-xs bg-background border border-input hover:bg-accent/60 hover:text-accent-foreground data-[state=on]:bg-accent data-[state=on]:text-accent-foreground"
                      >
                        Original
                      </ToggleGroupItem>
                    )}
                    <ToggleGroupItem 
                      value="cfa" 
                      aria-label="Show CFA" 
                      className="flex-1 text-xs bg-background border border-input hover:bg-accent/60 hover:text-accent-foreground data-[state=on]:bg-accent data-[state=on]:text-accent-foreground"
                    >
                      CFA
                    </ToggleGroupItem>
                    <ToggleGroupItem 
                      value="reconstruction" 
                      aria-label="Show reconstruction" 
                      className="flex-1 text-xs bg-background border border-input hover:bg-accent/60 hover:text-accent-foreground data-[state=on]:bg-accent data-[state=on]:text-accent-foreground"
                    >
                      Reconstruction
                    </ToggleGroupItem>
                  </ToggleGroup>
                </div>

                <div className="space-y-4 pt-2 border-t">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium flex items-center gap-2">
                      <Columns className="w-3 h-3" />
                      Comparison Mode
                    </label>
                    <Switch checked={comparisonMode} onCheckedChange={setComparisonMode} />
                  </div>
                  {comparisonMode && (
                    <div className="space-y-3 animate-in fade-in slide-in-from-top-2">
                      <div className="space-y-2">
                        <label className="text-xs font-medium">Layout</label>
                        <Select value={comparisonLayout} onValueChange={(v: 'side-by-side' | '4-up') => setComparisonLayout(v)}>
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="side-by-side">Side-by-Side</SelectItem>
                            <SelectItem value="4-up">4-Up</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <label className="text-xs font-medium">Comparison Preset</label>
                        <Select 
                          value={comparisonPreset} 
                          onValueChange={(v: ComparisonPreset) => {
                            setComparisonPreset(v);
                            if (v !== 'custom') {
                              // Auto-update configs when preset changes
                              const configs = getPresetConfigs(v, comparisonLayout, !!hasGroundTruth);
                              if (comparisonLayout === 'side-by-side') {
                                setCustomViewportConfigs(configs.slice(0, 2));
                              } else {
                                setCustomViewportConfigs(configs);
                              }
                            }
                          }}
                        >
                          <SelectTrigger><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="original-vs-reconstruction">Original vs Reconstruction</SelectItem>
                            <SelectItem value="cfa-vs-reconstruction">CFA vs Reconstruction</SelectItem>
                            <SelectItem value="algorithm-comparison">Algorithm Comparison</SelectItem>
                            <SelectItem value="cfa-comparison">CFA Comparison</SelectItem>
                            <SelectItem value="algorithm-cfa-comparison">Algorithm on Different CFAs</SelectItem>
                            <SelectItem value="4-up-standard">4-Up Standard</SelectItem>
                            <SelectItem value="custom">Custom</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      {comparisonPreset === 'custom' && (
                        <div className="space-y-3 pt-2 border-t animate-in fade-in slide-in-from-top-2">
                          <label className="text-xs font-medium">Custom Viewport Configuration</label>
                          {(comparisonLayout === 'side-by-side' ? customViewportConfigs.slice(0, 2) : customViewportConfigs).map((config, idx) => (
                            <div key={idx} className="space-y-2 p-2 bg-muted/20 rounded border border-border/50">
                              <div className="text-[10px] font-medium text-muted-foreground">Viewport {idx + 1}</div>
                              <div className="space-y-2">
                                <Select
                                  value={config.viewType}
                                  onValueChange={(v: 'original' | 'cfa' | 'reconstruction') => {
                                    const newConfigs = [...customViewportConfigs];
                                    newConfigs[idx] = { ...config, viewType: v };
                                    setCustomViewportConfigs(newConfigs);
                                  }}
                                >
                                  <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                                  <SelectContent>
                                    {hasGroundTruth && <SelectItem value="original">Original</SelectItem>}
                                    <SelectItem value="cfa">CFA</SelectItem>
                                    <SelectItem value="reconstruction">Reconstruction</SelectItem>
                                  </SelectContent>
                                </Select>
                                {config.viewType === 'cfa' && (
                                  <Select
                                    value={config.cfaPattern || 'bayer'}
                                    onValueChange={(v: CFAType) => {
                                      const newConfigs = [...customViewportConfigs];
                                      newConfigs[idx] = { ...config, cfaPattern: v };
                                      setCustomViewportConfigs(newConfigs);
                                    }}
                                  >
                                    <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="bayer">Bayer</SelectItem>
                                      <SelectItem value="xtrans">X-Trans</SelectItem>
                                    </SelectContent>
                                  </Select>
                                )}
                                {config.viewType === 'reconstruction' && (
                                  <>
                                    <Select
                                      value={config.useAlgorithm2 ? 'algorithm2' : 'algorithm1'}
                                      onValueChange={(v: 'algorithm1' | 'algorithm2') => {
                                        const newConfigs = [...customViewportConfigs];
                                        newConfigs[idx] = { ...config, useAlgorithm2: v === 'algorithm2' };
                                        setCustomViewportConfigs(newConfigs);
                                      }}
                                    >
                                      <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                                      <SelectContent>
                                        <SelectItem value="algorithm1">Algorithm A ({getAlgorithmName(algorithm)})</SelectItem>
                                        <SelectItem value="algorithm2">Algorithm B ({getAlgorithmName(algorithm2)})</SelectItem>
                                      </SelectContent>
                                    </Select>
                                    <Select
                                      value={config.cfaPattern || cfaType}
                                      onValueChange={(v: CFAType) => {
                                        const newConfigs = [...customViewportConfigs];
                                        newConfigs[idx] = { ...config, cfaPattern: v };
                                        setCustomViewportConfigs(newConfigs);
                                      }}
                                    >
                                      <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="CFA Pattern" /></SelectTrigger>
                                      <SelectContent>
                                        <SelectItem value="bayer">Bayer CFA</SelectItem>
                                        <SelectItem value="xtrans">X-Trans CFA</SelectItem>
                                        <SelectItem value="foveon">Foveon CFA</SelectItem>
                                      </SelectContent>
                                    </Select>
                                  </>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
          
          {/* Center Panel: Canvas */}
          <div className="lg:col-span-6 xl:col-span-5 flex flex-col h-full min-w-0 overflow-hidden">
            <Card className="h-full flex flex-col border-border shadow-sm bg-card overflow-hidden">
              <div className={`flex-1 relative min-w-0 min-h-0 overflow-hidden ${
                comparisonMode && comparisonLayout === 'side-by-side' 
                  ? 'grid grid-rows-2 divide-y divide-border' 
                  : comparisonMode && comparisonLayout === '4-up'
                  ? 'grid grid-cols-2 grid-rows-2 divide-x divide-y divide-border'
                  : ''
              }`}>
                 {!comparisonMode ? (
                   // Single viewport mode
                   createViewport(viewport1Ref, 1, { viewType: viewMode, cfaPattern: viewMode === 'cfa' ? cfaType : undefined })
                 ) : (() => {
                   // Get viewport configs based on preset or custom
                   const viewportConfigs = comparisonPreset === 'custom' 
                     ? customViewportConfigs 
                     : getPresetConfigs(comparisonPreset, comparisonLayout, !!hasGroundTruth);
                   
                   const viewportRefs = [viewport1Ref, viewport2Ref, viewport3Ref, viewport4Ref];
                   const numViewports = comparisonLayout === 'side-by-side' ? 2 : 4;
                   
                   return (
                     <>
                       {viewportConfigs.slice(0, numViewports).map((config, idx) => 
                         React.cloneElement(
                           createViewport(viewportRefs[idx], (idx + 1) as 1 | 2 | 3 | 4, config),
                           { key: idx }
                         )
                       )}
                     </>
                   );
                 })()}

              {showProcessingOverlay && (
                <div className="pointer-events-none absolute bottom-3 right-3 z-40 w-[360px] max-w-[80vw]">
                  <div className="pointer-events-auto bg-card/95 border-2 border-primary/50 shadow-2xl rounded-xl px-4 py-3 flex items-start gap-4 backdrop-blur-sm">
                    <Loader2 className="w-6 h-6 mt-0.5 text-primary animate-spin shrink-0" />
                    <div className="flex-1 space-y-2">
                      <div className="text-sm font-semibold text-foreground tracking-tight">Running demosaicing…</div>
                      <div className="text-[12px] text-muted-foreground leading-snug">{processingLabel}</div>
                      <div className="relative h-3 w-full overflow-hidden rounded-full bg-muted">
                        <div className="absolute inset-y-0 left-0 w-1/2 bg-gradient-to-r from-primary/70 via-primary to-primary/70 animate-progress-indeterminate" />
                      </div>
                    </div>
                  </div>
                </div>
              )}
              </div>
              
              <div className="h-12 border-t bg-card flex items-center justify-between px-4 shrink-0">
                 <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">Zoom:</span>
                    <Button variant="ghost" size="icon" className="h-8 w-8" onClick={handleZoomOut}><ZoomOut className="w-4 h-4" /></Button>
                    <Button variant="ghost" size="sm" className="h-8 font-mono w-16 text-center text-xs" onClick={toggleFit}>
                       {isFit ? "FIT" : `${Math.round(zoom * 100)}%`}
                    </Button>
                    <Button variant="ghost" size="icon" className="h-8 w-8" onClick={handleZoomIn}><ZoomIn className="w-4 h-4" /></Button>
                 </div>
                 {hoverPos && <div className="font-mono text-xs text-muted-foreground">x: {hoverPos.x}, y: {hoverPos.y}</div>}
              </div>
            </Card>
          </div>

          {/* Right Panel: Math & Inspector */}
          <div className="lg:col-span-3 xl:col-span-4 h-full overflow-y-auto min-w-0">
            {/* If comparison mode, show condensed traces or tabs? For simplicity, show Tab 1 / Tab 2 */}
            {comparisonMode ? (
                <div className="space-y-4">
                    <DemosaicMathExplanation 
                      cfaType={cfaType}
                      algorithm={algorithm}
                      x={selectedPos?.x}
                      y={selectedPos?.y}
                      error={selectedPos && errorStats && (input?.mode === 'lab' || input?.mode === 'synthetic') ? {
                        rgb: { r: 0, g: 0, b: 0 },
                        l2: errorStats.l2Map ? errorStats.l2Map[selectedPos.y * input!.width + selectedPos.x] : 0
                      } : undefined}
                      errorStats={errorStats}
                      input={input}
                      params={params}
                    />
                    <div className="text-center text-xs font-medium pt-2 border-t">Algorithm B</div>
                    <DemosaicMathExplanation 
                      cfaType={cfaType}
                      algorithm={algorithm2}
                      x={selectedPos?.x}
                      y={selectedPos?.y}
                      error={selectedPos && errorStats2 && (input?.mode === 'lab' || input?.mode === 'synthetic') ? {
                        rgb: { r: 0, g: 0, b: 0 },
                        l2: errorStats2.l2Map ? errorStats2.l2Map[selectedPos.y * input!.width + selectedPos.x] : 0
                      } : undefined}
                      errorStats={errorStats2}
                      input={input}
                      params={params2}
                    />
                </div>
            ) : (
                <DemosaicMathExplanation 
                  cfaType={cfaType}
                  algorithm={algorithm}
                  x={selectedPos?.x}
                  y={selectedPos?.y}
                  error={selectedPos && errorStats && (input?.mode === 'lab' || input?.mode === 'synthetic') ? {
                    rgb: { r: 0, g: 0, b: 0 },
                    l2: errorStats.l2Map ? errorStats.l2Map[selectedPos.y * input!.width + selectedPos.x] : 0
                  } : undefined}
                  errorStats={errorStats}
                  input={input}
                  syntheticType={syntheticType}
                  params={params}
                />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
