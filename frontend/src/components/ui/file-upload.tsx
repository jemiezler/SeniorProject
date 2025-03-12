"use client";

import React, { useRef, useState, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { motion } from "framer-motion";
import { IconUpload, IconX } from "@tabler/icons-react";
import { Alert } from "@heroui/react";
import Button from "@/components/ui/Button";
import { useRouter } from "next/navigation";

export const FileUpload = ({
  onSubmit,
}: {
  onSubmit?: (files: File[]) => void;
}) => {
  const [files, setFiles] = useState<File[]>([]);
  const [alert, setAlert] = useState<{
    title: string;
    description: string;
    type: "success" | "error";
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter(); 
  const showAlert = (
    title: string,
    description: string,
    type: "success" | "error"
  ) => {
    setAlert({ title, description, type });
    setTimeout(() => setAlert(null), 3000);
  };

  const handleFileChange = (newFiles: File[]) => {
    const validFiles = newFiles.filter((file) =>
      ["image/png", "image/jpeg", "image/jpg"].includes(file.type)
    );

    const file = newFiles[0];
    const imageUrl = URL.createObjectURL(file);

    localStorage.setItem("uploadedImage", imageUrl);


    setFiles((prev) => [...prev, file]);
    showAlert("Upload Successful", "File uploaded successfully!", "success");

    if (validFiles.length === 0) {
      showAlert(
        "Invalid File Type",
        "Only PNG, JPEG, and JPG are allowed.",
        "error"
      );
      return;
    }

    const filteredFiles = validFiles.filter(
      (file) => !files.some((existingFile) => existingFile.name === file.name)
    );

    if (filteredFiles.length === 0) {
      showAlert("Duplicate Files", "Some files are already uploaded.", "error");
      return;
    }

    setFiles((prev) => [...prev, ...filteredFiles]);
    showAlert("Upload Successful", "Files uploaded successfully!", "success");
  };

  // Handle file removal
  const handleRemoveFile = (fileToRemove: File) => {
    setFiles((prev) => prev.filter((file) => file !== fileToRemove));
    showAlert(
      "File Removed",
      `${fileToRemove.name} has been removed.`,
      "error"
    );
  };

  // Prevent submission if no files exist
  const handleSubmit = () => {
    if (files.length === 0) {
      showAlert(
        "No Files Uploaded",
        "Please upload at least one file.",
        "error"
      );
      return;
    }

    onSubmit?.(files);
    showAlert(
      "Submission Successful",
      "Files submitted successfully!",
      "success"
    );

    // After submission, redirect to the result page
    router.push("/result");
  };

  // Clean up object URLs to prevent memory leaks
  useEffect(() => {
    return () => {
      files.forEach((file) => URL.revokeObjectURL(URL.createObjectURL(file)));
    };
  }, [files]);

  // Dropzone Config
  const { getRootProps, isDragActive } = useDropzone({
    multiple: true,
    accept: {
      "image/png": [".png"],
      "image/jpeg": [".jpeg", ".jpg"],
    },
    onDrop: handleFileChange,
  });

  return (
    <div className="w-full">
      {/* Alert Notification */}
      {alert && (
        <div className="mb-4">
          <Alert title={alert.title} description={alert.description} />
        </div>
      )}

      {/* Drag & Drop Container */}
      <div {...getRootProps()} className="cursor-pointer">
        <motion.div
          onClick={() => fileInputRef.current?.click()}
          whileHover={{ scale: 1.02 }}
          className="p-10 group block rounded-lg w-full relative overflow-hidden"
        >
          {/* Hidden File Input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/png, image/jpeg, image/jpg"
            onChange={(e) => handleFileChange(Array.from(e.target.files || []))}
            className="hidden"
          />

          <div className="flex flex-col items-center justify-center bg-bgGreen1 h-[400px] rounded-2xl border border-stroke1 border-dashed p-4 w-[932px]">
            <p className="text-gray-600 dark:text-gray-400">
              Drag and drop your images here
            </p>
            <p className="text-gray-500 dark:text-gray-500 text-sm mt-5">or</p>

            <div className="relative w-full mt-5 max-w-xl mx-auto flex">
              <div className="max-h-[200px] overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-700 w-full">
                {files.length > 0 ? (
                  files.map((file, idx) => (
                    <motion.div
                      key={file.name}
                      layoutId={`file-${idx}`}
                      className="relative bg-white dark:bg-neutral-900 flex items-center p-4 w-full rounded-md shadow-sm mb-2"
                    >
                      <img
                        src={URL.createObjectURL(file)}
                        alt={file.name}
                        className="w-16 h-16 object-cover rounded-md mr-4"
                      />

                      <div className="flex flex-col flex-1 min-w-0">
                        <motion.p
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="text-base text-neutral-700 dark:text-neutral-300 truncate max-w-[450px]"
                          title={file.name}
                        >
                          {file.name}
                        </motion.p>
                        <motion.p
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="text-sm text-neutral-600 dark:text-neutral-400"
                        >
                          {(file.size / (1024 * 1024)).toFixed(2)} MB
                        </motion.p>
                      </div>

                      {/* Remove File Button */}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveFile(file);
                        }}
                        className="ml-4 text-red-500 hover:text-red-700 flex-shrink-0"
                      >
                        <IconX className="w-5 h-5" />
                      </button>
                    </motion.div>
                  ))
                ) : (
                  <motion.div
                    layoutId="file-upload-placeholder"
                    className="relative group-hover:shadow-2xl bg-white dark:bg-neutral-900 flex items-center justify-center h-32 w-full max-w-[8rem] mx-auto rounded-md shadow-lg"
                  >
                    {isDragActive ? (
                      <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-neutral-600 flex flex-col items-center"
                      >
                        Drop it
                        <IconUpload className="h-4 w-4 text-neutral-600 dark:text-neutral-400" />
                      </motion.p>
                    ) : (
                      <IconUpload className="h-4 w-4 text-neutral-600 dark:text-neutral-300" />
                    )}
                  </motion.div>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      <div className="mt-6 text-center">
        <Button
          text="Check"
          onClick={handleSubmit}
          disabled={files.length === 0}
          className={`px-6 py-2 rounded-lg ${
            files.length === 0
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-green-600 hover:bg-green-700 text-white"
          }`}
        />
      </div>
    </div>
  );
};
