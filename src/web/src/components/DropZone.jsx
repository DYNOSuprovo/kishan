import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileImage } from 'lucide-react';
import { motion } from 'framer-motion';

const DropZone = ({ onFileAccepted, file }) => {
    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles?.length > 0) {
            onFileAccepted(acceptedFiles);
        }
    }, [onFileAccepted]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.jpeg', '.jpg', '.png']
        },
        maxFiles: 1,
        multiple: false
    });

    return (
        <div {...getRootProps()} className="w-full h-full flex flex-col items-center justify-center cursor-pointer outline-none z-10">
            <input {...getInputProps()} />

            <motion.div
                animate={{ scale: isDragActive ? 1.05 : 1 }}
                className="flex flex-col items-center text-center space-y-6"
            >
                <div className={`
          w-24 h-24 rounded-2xl flex items-center justify-center transition-all duration-300
          ${isDragActive ? 'bg-cyber-green text-black shadow-[0_0_30px_#00ff9d]' : 'bg-white/5 border border-white/10 text-cyber-green group-hover:border-cyber-green/50'}
        `}>
                    {file ? <FileImage className="w-10 h-10" /> : <Upload className="w-10 h-10" />}
                </div>

                <div className="space-y-2">
                    {file ? (
                        <>
                            <h3 className="text-xl font-bold text-white max-w-[200px] truncate">{file.name}</h3>
                            <p className="text-white/40 text-sm">Click or drag to replace</p>
                        </>
                    ) : (
                        <>
                            <h3 className="text-xl font-bold text-white">Upload Leaf Image</h3>
                            <p className="text-white/40 text-sm">Drag & drop or click to browse</p>
                            <p className="text-xs text-white/20 mt-4 font-mono">SUPPORTS: JPEG, PNG</p>
                        </>
                    )}
                </div>
            </motion.div>
        </div>
    );
};

export default DropZone;
