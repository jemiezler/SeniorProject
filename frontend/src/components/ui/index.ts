  
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