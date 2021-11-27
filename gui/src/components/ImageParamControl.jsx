/*
Message to the futurrrrrrrre:

Eventually we will want to unify the code that powers this application AND
https://github.com/allenai/prior-demo/ AND https://github.com/allenai/comet-demo

When we do, DON'T use the code below, rather pick code from the other 2 repos.
This code, while fine, props have drifted and its JSX not TSX.
*/

import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { Upload, message } from '@allenai/varnish';
import { LoadingOutlined, UploadOutlined } from '@ant-design/icons';

/* props = {
    modelParams: {
        imgSrc?: string;
        imageName?: string;
        image?: File;
    }
    onChange: (newModelParams) => void;
}*/
export const ImageParamControl = (props) => {
    const [localState, setLocalState] = useState({});
    const [imageLoading, setImageLoading] = useState(false);

    useEffect(
        () => {
            if (props.modelParams.imgSrc && props.modelParams.imgSrc !== localState.imgSrc) {
                fetchImage(props.modelParams.imgSrc);
            }
        },
        [props.modelParams.imgSrc]
    );

    const compressAndSubmit = (
        file,
        maxFileBytes,
        onSuccess,
        onError
    ) => {
        if (file.size > maxFileBytes) {
            compressImage({ file, maxFileBytes, onSuccess, onError });
        } else {
            onSuccess(file);
        }
    };

    const beforeUpload = (file) => {
        const isImage =
            file.type === 'image/bmp' ||
            file.type === 'image/gif' ||
            file.type === 'image/png' ||
            file.type === 'image/jpeg' ||
            file.type === 'image/tiff';
        if (!isImage) {
            message.error('You can only upload JPG/PNG/BMP/GIF/TIFF files');
        } else {
            setImageLoadingAndSendEvent(true);
        }
        return isImage;
    };

    const handleImageChange = (info) => {
        if (info.file.status === 'done') {
            const file = info.file.originFileObj;
            if (file) {
                compressAndSubmit(
                    file,
                    1024 * 1024, // compressing if larger than 1MB
                    (compressedFile) => {
                        setStateAndSendEvent({
                            imgSrc: URL.createObjectURL(compressedFile),
                            imageName: compressedFile.name,
                            image: compressedFile
                        });
                        setImageLoadingAndSendEvent(false);
                    },
                    () => {
                        message.error(`${info.file.name} file upload failed: ${info.file.error.message}`);
                        setStateAndSendEvent({
                            imgSrc: undefined,
                            imageName: undefined,
                            image: undefined
                        });
                        setImageLoadingAndSendEvent(false);
                    }
                );
            }
        } else if (info.file.status === 'error') {
            // If the image is too large, then info.file.error.message will
            // look like "cannot post api/permalink/noop 413'"
            const errorIs413 = info.file.error.message.match("^cannot post .* 413'$");

            const maxFileSize = 5 * 1024 * 1024;
            if (errorIs413 && info.file.size > maxFileSize) {
                // Show a friendly "too large" error if it's appropriate to do so
                message.error(`${info.file.name} file is too large; must be smaller than ${maxFileSize} bytes`);
            } else {
                // Otherwise, it's a different error.
                message.error(`${info.file.name} file upload failed: ${info.file.error.message}`);
            }

            setStateAndSendEvent({
                imgSrc: undefined,
                imageName: undefined,
                image: undefined
            });
            setImageLoadingAndSendEvent(false);
        }
    };

    async function fetchImage(imgSrc) {
        setImageLoadingAndSendEvent(true);
        let s = {
            imgSrc: imgSrc,
            imageName: undefined,
            image: undefined
        };
        if (imgSrc) {
            const response = await fetch(imgSrc);
            const blob = await response.blob();

            const file = blob; // convert blob to file
            file.lastModifiedDate = new Date();
            file.name = imgSrc;
            s = {
                imgSrc: imgSrc,
                imageName: file.name,
                image: file
            };
        }
        setStateAndSendEvent(s);
        setImageLoadingAndSendEvent(false);
    }

    const setStateAndSendEvent = (s) => {
        const val = { ...localState, ...s };
        setLocalState(val);
        props.onChange(val);
    };

    const setImageLoadingAndSendEvent = (imageLoading) => {
        setImageLoading(imageLoading);
    };

    return (
        <React.Fragment>
            <div title="Upload an Image">
                <Dragger
                    onChange={handleImageChange}
                    showUploadList={false}
                    // this is just a noop endpoint, we need an endpoint, but we dont need
                    // to save the image
                    action="api/permalink/noop"
                    beforeUpload={beforeUpload}>
                    {imageLoading ? (
                        <DraggerMessage>
                            <LoadingOutlined /> Loading...
                        </DraggerMessage>
                    ) : null}
                    {!imageLoading && localState.imgSrc ? (
                        <DraggerImg src={localState.imgSrc} />
                    ) : null}
                    {!imageLoading && !localState.imgSrc ? (
                        <DraggerMessage>
                            <UploadOutlined /> Upload an Image
                        </DraggerMessage>
                    ) : null}
                </Dragger>
            </div>
        </React.Fragment>
    );
};

// TODO: consider making a Promise<Image> instead of callback
// TODO: consider moving to a webworker
export const compressImage = ({
    file,
    maxFileBytes,
    newFileName,
    onSuccess,
    onError
}) => {
    const path = require('path');
    const originalFileName = path.basename(file.name);
    const fileName = newFileName || `${originalFileName}_c.jpg`;
    // console.log(`Compressing ${file.name} (${file.size}Bytes)`);
    // maxPixels captures the maximum pixels in an image that's maxFileBytes in size.
    // 3Bytes/px was calculated from random data case of png/jpg quality 1 https://superuser.com/questions/636333/what-is-the-largest-size-of-a-640x480-jpeg
    const maxPixels = maxFileBytes / 3;
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.src = event.target.result;
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const curPixels = img.width * img.height;
            const scale = Math.sqrt(maxPixels) / Math.sqrt(curPixels);
            canvas.width = img.width * scale;
            canvas.height = img.height * scale;
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                ctx.canvas.toBlob(
                    (blob) => {
                        const compressedFile = new File([blob], fileName, {
                            type: 'image/jpeg',
                            lastModified: Date.now()
                        });
                        // console.log(`Generated ${fileName} (${compressedFile.size}Bytes)`);
                        onSuccess(compressedFile);
                    },
                    'image/jpeg',
                    1
                );
            } else {
                if (onError) {
                    onError('Invalid context');
                }
            }
        };
        reader.onerror = (error) => {
            console.log(error);
            if (onError) {
                onError('Error compressing image.');
            }
        };
    };
    reader.readAsDataURL(file);
};

const Dragger = styled(Upload.Dragger)`
    &&& {
        .ant-upload-drag {
            height: 260px;
        }
        .ant-upload-btn {
            padding: 0;
        }
    }
`;

const DraggerMessage = styled.div`
    padding: ${({ theme }) => `${theme.spacing.md} 0`};
`;

const DraggerImg = styled.img`
    max-height: ${({ theme }) => `calc(267px - ${theme.spacing.xxs} - ${theme.spacing.xxs})`};
    max-width: ${({ theme }) => `calc(100% - ${theme.spacing.xxs} - ${theme.spacing.xxs})`};
    padding: ${({ theme }) => theme.spacing.xxs};
`;

export async function blobToString(blob) {
    return new Promise((resolve, reject) => {
        let reader = new FileReader();
        reader.onloadend = () => {
            var base64String = reader.result;
            // Base64 Encoded String without additional data: Attributes.
            resolve(base64String.substr(base64String.indexOf(',') + 1));
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    })
}
