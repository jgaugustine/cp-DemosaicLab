import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Input } from '@/components/ui/input';
import { BenchmarkSummary } from './benchmark/BenchmarkSummary';
import { BenchmarkTable } from './benchmark/BenchmarkTable';
import { BenchmarkCharts } from './benchmark/BenchmarkCharts';
import { ErrorHeatmap } from './benchmark/ErrorHeatmap';
import { BenchmarkResult, BenchmarkConfig, BenchmarkProgress } from '@/types/benchmark';
import { DemosaicInput, DemosaicAlgorithm, CFAType, DemosaicParams } from '@/types/demosaic';
import { runSingleBenchmark } from '@/lib/benchmark';
import { simulateCFA } from '@/lib/cfa';
import {
  createZonePlate,
  createFineCheckerboard,
  createColorSweep,
  createStarburst,
  createDiagonalLines,
  createSineWaveGratings,
  createColorPatches,
  createColorFringes,
} from '@/lib/synthetic';
import { Play, Download, FileJson, FileSpreadsheet, X, Image as ImageIcon, Trash2, Pencil, CheckSquare, Square } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { downsizeImageToDataURL } from '@/lib/imageResize';

interface BenchmarkModeProps {
  uploadedImages?: Map<string, ImageData>; // Optional: uploaded images for benchmarking
  defaultParams?: DemosaicParams;
}

const DEFAULT_WIDTH = 512;
const DEFAULT_HEIGHT = 512;

const SYNTHETIC_PATTERNS = [
  { id: 'zoneplate', name: 'Zone Plate', create: createZonePlate },
  { id: 'checker', name: 'Fine Checkerboard', create: createFineCheckerboard },
  { id: 'sweep', name: 'Color Sweep', create: createColorSweep },
  { id: 'star', name: 'Starburst', create: createStarburst },
  { id: 'diagonal', name: 'Diagonal Lines', create: createDiagonalLines },
  { id: 'sine', name: 'Sine Wave Gratings', create: createSineWaveGratings },
  { id: 'patches', name: 'Color Patches', create: createColorPatches },
  { id: 'fringes', name: 'Color Fringes', create: createColorFringes },
];

const ALL_ALGORITHMS: DemosaicAlgorithm[] = [
  'nearest',
  'bilinear',
  'niu_edge_sensing',
  'lien_edge_based',
  'wu_polynomial',
  'kiku_residual',
];

const ALL_CFA_PATTERNS: CFAType[] = ['bayer', 'xtrans'];

export function BenchmarkMode({ uploadedImages = new Map(), defaultParams }: BenchmarkModeProps) {
  const [config, setConfig] = useState<BenchmarkConfig>({
    algorithms: [...ALL_ALGORITHMS],
    testImages: SYNTHETIC_PATTERNS.map(p => p.id),
    cfaPatterns: ['bayer'],
    iterations: 3,
    enableQualityMetrics: true,
    testImageWidth: DEFAULT_WIDTH,
    testImageHeight: DEFAULT_HEIGHT,
  });

  const [progress, setProgress] = useState<BenchmarkProgress>({
    current: 0,
    total: 0,
    currentTest: '',
    isRunning: false,
    results: [],
  });

  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [groundTruthImages, setGroundTruthImages] = useState<Map<string, ImageData>>(new Map());
  const [customUploadedImages, setCustomUploadedImages] = useState<Map<string, ImageData>>(new Map());
  const [imageNames, setImageNames] = useState<Map<string, string>>(new Map());
  const [renamingImageId, setRenamingImageId] = useState<string | null>(null);
  const [renameInputValue, setRenameInputValue] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cancelRef = useRef(false);

  // Handle image upload
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
        const imageId = `uploaded_${Date.now()}_${file.name}`;
        setCustomUploadedImages(prev => {
          const newMap = new Map(prev);
          newMap.set(imageId, imageData);
          return newMap;
        });
        // Add to test images list
        setConfig(prev => ({
          ...prev,
          testImages: [...prev.testImages, imageId],
        }));
      };
      img.src = dataUrl;
    } catch (err) {
      console.error('Failed to upload image:', err);
    }
    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeCustomImage = (imageId: string) => {
    setCustomUploadedImages(prev => {
      const newMap = new Map(prev);
      newMap.delete(imageId);
      return newMap;
    });
    setImageNames(prev => {
      const newMap = new Map(prev);
      newMap.delete(imageId);
      return newMap;
    });
    setConfig(prev => ({
      ...prev,
      testImages: prev.testImages.filter(id => id !== imageId),
    }));
  };

  const startRename = (imageId: string) => {
    const currentName = imageNames.get(imageId) || imageId.replace(/^uploaded_\d+_/, '');
    setRenameInputValue(currentName);
    setRenamingImageId(imageId);
  };

  const saveRename = () => {
    if (renamingImageId && renameInputValue.trim()) {
      setImageNames(prev => {
        const newMap = new Map(prev);
        newMap.set(renamingImageId, renameInputValue.trim());
        return newMap;
      });
      setRenamingImageId(null);
      setRenameInputValue('');
    }
  };

  const cancelRename = () => {
    setRenamingImageId(null);
    setRenameInputValue('');
  };

  const getImageDisplayName = (imageId: string): string => {
    return imageNames.get(imageId) || imageId.replace(/^uploaded_\d+_/, '');
  };

  // Merge custom uploaded images with prop images
  const allUploadedImages = useMemo(() => {
    const merged = new Map(customUploadedImages);
    uploadedImages.forEach((value, key) => {
      merged.set(key, value);
    });
    return merged;
  }, [customUploadedImages, uploadedImages]);

  // Generate all test inputs
  const generateTestInputs = useCallback((cfg: BenchmarkConfig): Array<{
    input: DemosaicInput;
    imageName: string;
    groundTruth: ImageData;
  }> => {
    const inputs: Array<{ input: DemosaicInput; imageName: string; groundTruth: ImageData }> = [];
    const width = cfg.testImageWidth ?? DEFAULT_WIDTH;
    const height = cfg.testImageHeight ?? DEFAULT_HEIGHT;

    // Generate synthetic patterns
    for (const patternId of cfg.testImages) {
      if (patternId.startsWith('uploaded_')) {
        // Handle uploaded images
        const imageData = allUploadedImages.get(patternId);
        if (imageData) {
          for (const cfaPattern of cfg.cfaPatterns) {
            const cfa = simulateCFA(imageData, cfaPattern, 'RGGB');
            inputs.push({
              input: {
                mode: 'lab',
                groundTruthRGB: imageData,
                cfaPattern,
                cfaPatternMeta: cfaPattern === 'bayer' 
                  ? { tileW: 2, tileH: 2, layout: 'RGGB' }
                  : cfaPattern === 'xtrans'
                  ? { tileW: 6, tileH: 6, layout: 'xtrans' }
                  : { tileW: 1, tileH: 1, layout: '' },
                cfaData: cfa,
                width: imageData.width,
                height: imageData.height,
              },
              imageName: imageNames.get(patternId) || getImageDisplayName(patternId),
              groundTruth: imageData,
            });
          }
        }
      } else {
        // Generate synthetic pattern
        const pattern = SYNTHETIC_PATTERNS.find(p => p.id === patternId);
        if (pattern) {
          const imageData = pattern.create(width, height);
          for (const cfaPattern of cfg.cfaPatterns) {
            const cfa = simulateCFA(imageData, cfaPattern, 'RGGB');
            inputs.push({
              input: {
                mode: 'synthetic',
                groundTruthRGB: imageData,
                cfaPattern,
                cfaPatternMeta: cfaPattern === 'bayer' 
                  ? { tileW: 2, tileH: 2, layout: 'RGGB' }
                  : cfaPattern === 'xtrans'
                  ? { tileW: 6, tileH: 6, layout: 'xtrans' }
                  : { tileW: 1, tileH: 1, layout: '' },
                cfaData: cfa,
                width,
                height,
              },
              imageName: pattern.name,
              groundTruth: imageData,
            });
          }
        }
      }
    }

    return inputs;
  }, [allUploadedImages, imageNames]);

  const runBenchmark = useCallback(async () => {
    cancelRef.current = false;
    const testInputs = generateTestInputs(config);
    const totalTests = config.algorithms.length * testInputs.length;

    // Store ground truth images
    const groundTruthMap = new Map<string, ImageData>();
    testInputs.forEach(({ imageName, groundTruth }) => {
      groundTruthMap.set(imageName, groundTruth);
    });
    setGroundTruthImages(groundTruthMap);

    setProgress({
      current: 0,
      total: totalTests,
      currentTest: 'Starting benchmark...',
      isRunning: true,
      results: [],
    });

    setResults([]);
    const newResults: BenchmarkResult[] = [];
    let currentTest = 0;

    algorithmLoop: for (const algo of config.algorithms) {
      if (cancelRef.current) break algorithmLoop;

      for (const { input, imageName } of testInputs) {
        if (cancelRef.current) break algorithmLoop;

        const testName = `${algo} on ${imageName} with ${input.cfaPattern}`;
        setProgress(prev => ({
          ...prev,
          current: currentTest,
          currentTest: testName,
        }));

        try {
          const result = await runSingleBenchmark(
            input,
            algo,
            imageName,
            config.iterations,
            config.enableQualityMetrics,
            defaultParams,
            () => cancelRef.current // Pass cancellation check function
          );
          newResults.push(result);
          setResults([...newResults]);
        } catch (error) {
          if (error instanceof Error && error.message === 'Benchmark cancelled') {
            // Cancellation is expected, break from both loops
            break algorithmLoop;
          }
          console.error(`Benchmark failed for ${testName}:`, error);
        }
        
        // Check for cancellation after each test
        if (cancelRef.current) break algorithmLoop;

        currentTest++;
        
        // Yield to UI after each test to allow cancellation to be checked
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    // Only update final state if not cancelled
    if (!cancelRef.current) {
      setProgress(prev => ({
        ...prev,
        current: totalTests,
        currentTest: 'Complete',
        isRunning: false,
      }));
      setResults(newResults);
    } else {
      // Update progress to show cancellation
      setProgress(prev => ({
        ...prev,
        currentTest: 'Cancelled',
        isRunning: false,
      }));
      // Keep partial results if any were collected
      if (newResults.length > 0) {
        setResults(newResults);
      }
    }
  }, [config, generateTestInputs, defaultParams]);

  const handleCancel = () => {
    cancelRef.current = true;
    setProgress(prev => ({ ...prev, isRunning: false }));
  };

  const exportToCSV = () => {
    const headers = [
      'Algorithm',
      'Image',
      'CFA Pattern',
      'Width',
      'Height',
      'Pixel Count',
      'Avg Time (ms)',
      'Median Time (ms)',
      'Min Time (ms)',
      'Max Time (ms)',
      'Std Dev Time (ms)',
      'Throughput (MP/s)',
      'PSNR Total (dB)',
      'SSIM',
      'MSE Total',
    ];

    const rows = results.map(r => [
      r.algorithm,
      r.imageName,
      r.cfaPattern,
      r.width,
      r.height,
      r.pixelCount,
      r.performance.averageTimeMs.toFixed(4),
      r.performance.medianTimeMs.toFixed(4),
      r.performance.minTimeMs.toFixed(4),
      r.performance.maxTimeMs.toFixed(4),
      r.performance.stdDevTimeMs.toFixed(4),
      r.performance.throughputMPs.toFixed(4),
      r.quality?.psnr.total.toFixed(4) ?? 'N/A',
      r.quality?.ssim.toFixed(4) ?? 'N/A',
      r.quality?.mse.total.toFixed(4) ?? 'N/A',
    ]);

    const csv = [
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(',')),
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `benchmark_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportToJSON = () => {
    const json = JSON.stringify({
      timestamp: new Date().toISOString(),
      config,
      results,
      metadata: {
        browser: navigator.userAgent,
        platform: navigator.platform,
      },
    }, null, 2);

    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `benchmark_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const toggleAlgorithm = (algo: DemosaicAlgorithm) => {
    setConfig(prev => ({
      ...prev,
      algorithms: prev.algorithms.includes(algo)
        ? prev.algorithms.filter(a => a !== algo)
        : [...prev.algorithms, algo],
    }));
  };

  const toggleTestImage = (imageId: string) => {
    setConfig(prev => ({
      ...prev,
      testImages: prev.testImages.includes(imageId)
        ? prev.testImages.filter(id => id !== imageId)
        : [...prev.testImages, imageId],
    }));
  };

  const toggleCFA = (cfa: CFAType) => {
    setConfig(prev => ({
      ...prev,
      cfaPatterns: prev.cfaPatterns.includes(cfa)
        ? prev.cfaPatterns.filter(c => c !== cfa)
        : [...prev.cfaPatterns, cfa],
    }));
  };

  const checkAllAlgorithms = () => {
    setConfig(prev => ({
      ...prev,
      algorithms: [...ALL_ALGORITHMS],
    }));
  };

  const uncheckAllAlgorithms = () => {
    setConfig(prev => ({
      ...prev,
      algorithms: [],
    }));
  };

  const checkAllTestImages = () => {
    const allTestImages = [
      ...SYNTHETIC_PATTERNS.map(p => p.id),
      ...Array.from(customUploadedImages.keys()),
    ];
    setConfig(prev => ({
      ...prev,
      testImages: allTestImages,
    }));
  };

  const uncheckAllTestImages = () => {
    setConfig(prev => ({
      ...prev,
      testImages: [],
    }));
  };

  return (
    <div className="space-y-6 p-6">
      <Card>
        <CardHeader>
          <CardTitle>Benchmark Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <div className="flex items-center justify-between mb-3">
              <Label>Algorithms</Label>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={checkAllAlgorithms}
                  className="h-7 text-xs"
                >
                  <CheckSquare className="mr-1 h-3 w-3" />
                  Check All
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={uncheckAllAlgorithms}
                  className="h-7 text-xs"
                >
                  <Square className="mr-1 h-3 w-3" />
                  Uncheck All
                </Button>
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {ALL_ALGORITHMS.map(algo => (
                <div key={algo} className="flex items-center space-x-2">
                  <Checkbox
                    id={`algo-${algo}`}
                    checked={config.algorithms.includes(algo)}
                    onCheckedChange={() => toggleAlgorithm(algo)}
                  />
                  <Label htmlFor={`algo-${algo}`} className="font-normal cursor-pointer">
                    {algo.replace(/_/g, ' ')}
                  </Label>
                </div>
              ))}
            </div>
          </div>

          <Separator />

          <div>
            <div className="flex items-center justify-between mb-3">
              <Label>Test Images</Label>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={checkAllTestImages}
                  className="h-7 text-xs"
                >
                  <CheckSquare className="mr-1 h-3 w-3" />
                  Check All
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={uncheckAllTestImages}
                  className="h-7 text-xs"
                >
                  <Square className="mr-1 h-3 w-3" />
                  Uncheck All
                </Button>
              </div>
            </div>
            <div className="space-y-3">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {SYNTHETIC_PATTERNS.map(pattern => (
                  <div key={pattern.id} className="flex items-center space-x-2">
                    <Checkbox
                      id={`image-${pattern.id}`}
                      checked={config.testImages.includes(pattern.id)}
                      onCheckedChange={() => toggleTestImage(pattern.id)}
                    />
                    <Label htmlFor={`image-${pattern.id}`} className="font-normal cursor-pointer">
                      {pattern.name}
                    </Label>
                  </div>
                ))}
              </div>
              
              {/* Custom uploaded images */}
              {Array.from(customUploadedImages.keys()).length > 0 && (
                <div className="space-y-2 pt-2 border-t">
                  <Label className="text-sm font-medium">Uploaded Images</Label>
                  <div className="space-y-1">
                    {Array.from(customUploadedImages.keys()).map(imageId => (
                      <div key={imageId} className="flex items-center justify-between p-2 bg-muted rounded">
                        <div className="flex items-center space-x-2 flex-1 min-w-0">
                          <Checkbox
                            id={`image-${imageId}`}
                            checked={config.testImages.includes(imageId)}
                            onCheckedChange={() => toggleTestImage(imageId)}
                          />
                          <Label htmlFor={`image-${imageId}`} className="font-normal cursor-pointer text-sm truncate">
                            {getImageDisplayName(imageId)}
                          </Label>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => startRename(imageId)}
                            className="h-6 w-6 p-0"
                            title="Rename image"
                          >
                            <Pencil className="h-3 w-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => removeCustomImage(imageId)}
                            className="h-6 w-6 p-0"
                            title="Remove image"
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Upload button */}
              <Button
                variant="outline"
                className="w-full border-dashed"
                onClick={() => fileInputRef.current?.click()}
              >
                <ImageIcon className="mr-2 h-4 w-4" />
                Upload Custom Image
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleImageUpload}
              />
            </div>
          </div>

          <Separator />

          <div>
            <Label className="mb-3 block">CFA Patterns</Label>
            <div className="flex gap-4">
              {ALL_CFA_PATTERNS.map(cfa => (
                <div key={cfa} className="flex items-center space-x-2">
                  <Checkbox
                    id={`cfa-${cfa}`}
                    checked={config.cfaPatterns.includes(cfa)}
                    onCheckedChange={() => toggleCFA(cfa)}
                  />
                  <Label htmlFor={`cfa-${cfa}`} className="font-normal cursor-pointer uppercase">
                    {cfa}
                  </Label>
                </div>
              ))}
            </div>
          </div>

          <Separator />

          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="iterations">Iterations per test</Label>
              <Input
                id="iterations"
                type="number"
                min={1}
                max={10}
                value={config.iterations}
                onChange={(e) => setConfig(prev => ({ ...prev, iterations: parseInt(e.target.value) || 1 }))}
              />
            </div>
            <div>
              <Label htmlFor="enable-quality" className="flex items-center space-x-2 cursor-pointer">
                <Checkbox
                  id="enable-quality"
                  checked={config.enableQualityMetrics}
                  onCheckedChange={(checked) =>
                    setConfig(prev => ({ ...prev, enableQualityMetrics: checked === true }))
                  }
                />
                <span>Enable Quality Metrics</span>
              </Label>
            </div>
          </div>

          <div className="flex gap-2">
            <Button
              onClick={runBenchmark}
              disabled={progress.isRunning || config.algorithms.length === 0 || config.testImages.length === 0}
            >
              <Play className="mr-2 h-4 w-4" />
              Run Benchmark
            </Button>
            {progress.isRunning && (
              <Button variant="outline" onClick={handleCancel}>
                <X className="mr-2 h-4 w-4" />
                Cancel
              </Button>
            )}
          </div>

          {progress.isRunning && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>{progress.currentTest}</span>
                <span>{progress.current} / {progress.total}</span>
              </div>
              <Progress value={(progress.current / progress.total) * 100} />
            </div>
          )}
        </CardContent>
      </Card>

      {results.length > 0 && (
        <>
          <BenchmarkSummary results={results} />

          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle>Results</CardTitle>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={exportToCSV}>
                    <FileSpreadsheet className="mr-2 h-4 w-4" />
                    Export CSV
                  </Button>
                  <Button variant="outline" size="sm" onClick={exportToJSON}>
                    <FileJson className="mr-2 h-4 w-4" />
                    Export JSON
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="table" className="w-full">
                <TabsList>
                  <TabsTrigger value="table">Table</TabsTrigger>
                  <TabsTrigger value="charts">Charts</TabsTrigger>
                  <TabsTrigger value="heatmaps">Error Heatmaps</TabsTrigger>
                </TabsList>
                <TabsContent value="table" className="mt-4">
                  <BenchmarkTable results={results} />
                </TabsContent>
                <TabsContent value="charts" className="mt-4">
                  <BenchmarkCharts results={results} />
                </TabsContent>
                <TabsContent value="heatmaps" className="mt-4">
                  <ErrorHeatmap results={results} groundTruthImages={groundTruthImages} />
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </>
      )}

      {/* Rename Dialog */}
      <Dialog open={renamingImageId !== null} onOpenChange={(open) => !open && cancelRename()}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename Image</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <Label htmlFor="rename-input">Image Name</Label>
              <Input
                id="rename-input"
                value={renameInputValue}
                onChange={(e) => setRenameInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    saveRename();
                  } else if (e.key === 'Escape') {
                    cancelRename();
                  }
                }}
                autoFocus
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={cancelRename}>
              Cancel
            </Button>
            <Button onClick={saveRename} disabled={!renameInputValue.trim()}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

