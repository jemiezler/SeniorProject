export interface AnalysisResult {
    prediction: string;
    confidence: number;
    segmentation: {
      url: string;
      width: number;
      height: number;
    };
    recommendations: string[];
  }
  
  export interface FileUploadProps {
    onFileSelect: (files: File[]) => void;
    onRemoveFile: (index: number) => void;
    selectedFiles: File[];
    previews: string[];
    isAnalyzing: boolean;
    progress: number;
  }
  
  export interface LoadingProps {
    progress: number;
    message?: string;
  } 