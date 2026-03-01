"""
Used to annotate the dataset for Gradio interface.

Usage:
python gradio_filter_and_annotation.py --base_dir <split_dir> --save_ann_file <annotate_file>

Args:
    --base_dir: The root directory path containing all page directories.
    --save_ann_file: The file name to save annotations.

Directory Description
<split_dir>/
├── candidate/
│   ├── page_stats.json
│   ├── filekey1/
│   │   ├── page_id1.png
│   │   ├── page_id2.png
│   │   └── ...
│   ├── filekey2/
│   │   ├── page_id1.png
│   │   ├── page_id2.png
│   │   └── ...
│   └── ...
├── selected/
│   ├── filekey2/
│   │   ├── page_id1.png
│   │   ├── page_id2.png
│   │   └── ...
│   └── ...
├── trash/
│   └── filekey3/...
├── report.json # Statistics
└── final_annotation.json # Final annotation file
"""

import gradio as gr
import json
import os
import argparse
from glob import glob
import shutil
import socket

class AnnotationValidator:
    def __init__(self, base_dir, save_ann_file='final_annotation.json'):
        """
        Initialize the annotation validator.
        
        Args:
            base_dir (str): The root directory path containing all page directories.
            ref_ann_file (str): The reference annotation file name.
            save_ann_file (str): The file name to save annotations.
        """

        # Directory setup
        self.base_dir = base_dir
        self.candidate_dir = os.path.join(base_dir, "candidate")
        self.trash_dir = os.path.join(base_dir, "trash")
        self.selected_dir = os.path.join(base_dir, "selected")
        if not os.path.exists(self.candidate_dir):
            raise ValueError(f"Candidate directory does not exist: {self.candidate_dir}")
        os.makedirs(self.trash_dir, exist_ok=True)
        os.makedirs(self.selected_dir, exist_ok=True)

        # For quick page index lookup
        self.page_idx ={} # {<filekey>:{<page_id>: <page_index>}}
        
        # Image paths and annotation file name
        self.page_list = [] # (<filekey>, <page_id>)
        self.img_paths= {} #{<filekey>:{<page_id>: <image_path>}}
        self.save_ann_file = save_ann_file
        for i, img_path in enumerate(sorted(glob(os.path.join(self.candidate_dir, "**", "*.png"), recursive=True))):
            filekey = os.path.basename(os.path.dirname(img_path))
            # Some pages use : as a separator, which needs to be replaced with _
            page_id = os.path.basename(img_path)[:-4].replace('_', ':').replace('-', ';')
            self.page_list.append((filekey, page_id))
            if filekey not in self.img_paths:
                self.img_paths[filekey] = {}
                self.page_idx[filekey] = {}
            self.img_paths[filekey][page_id] = img_path
            self.page_idx[filekey][page_id] = i
        # Convenient for checking the last page
        self.last_page_index = len(self.page_list) - 1
        
        # Define annotation attribute configuration
        self.annotation_fields = {
            'keep': {
                'type': 'choice',
                'options': ['yes', 'no'],
            },
            'platform': {
                'type': 'choice',
                'options': ['mobile', 'desktop'],
            },
            'complexity': {
                'type': 'choice',
                'options': ['low', 'mid', 'high', 'hard'],
            },
            'quality_rating': {
                'type': 'choice',
                'options': [1, 2, 3],
            },
        }
        self.field_names = list(self.annotation_fields.keys())

        """ Annotation Cache
        {
            <filekey>: {
                <page_id>: <annotation_dict>
            }
        }
        """
        self.page_annotation_cache = {}

        # Convenient for saving only validated pages
        self.page_save_flag = [False] * len(self.page_list)
        
        # Initialize cache
        self.first_unsaved_page = self.last_page_index  # Index of the first unsaved page, defaults to the last page
        self.init_cache()

    def init_cache(self):
        """Initialize cache: first update from save_ann_file, and find the first unsaved page"""
        all_annotation_path = os.path.join(self.base_dir, self.save_ann_file)
        all_annotation = {}
        if os.path.exists(all_annotation_path):
            with open(all_annotation_path, 'r', encoding='utf-8') as f:
                all_annotation = json.load(f)

        # Initialize all page caches with None for each attribute
        for filekey, key_dict in self.img_paths.items():
            self.page_annotation_cache[filekey] = all_annotation.get(filekey, {})
            for page_id in key_dict.keys():
                self.page_annotation_cache[filekey][page_id] = {field_name: self.page_annotation_cache[filekey].get(page_id, {}).get(field_name, None) for field_name in self.field_names}
        
        # Modify cache based on saved annotation files
        for filekey in self.img_paths.keys():
            save_ann_path = os.path.join(self.candidate_dir, filekey, self.save_ann_file)
            if os.path.exists(save_ann_path):
                try:
                    with open(save_ann_path, 'r', encoding='utf-8') as f:
                        saved_data = json.load(f)
                        if saved_data:
                            # Update data in the cache
                            for page_id, page_annotation in saved_data.items():
                                for field_name in self.field_names:
                                    if field_name in page_annotation:
                                        self.page_annotation_cache[filekey][page_id][field_name] = page_annotation[field_name]
                                self.page_save_flag[self.page_idx[filekey][page_id]] = True
                except Exception as e:
                    print(f"Warning: Failed to read annotation file {save_ann_path}: {e}")
        
        # Find the first unsaved page and update the page save flag
        for i in range(len(self.page_list)):
            if not self.page_save_flag[i]:
                self.first_unsaved_page = i
                break
        
    @property
    def earliest_unsaved_page(self):
        """Find the earliest unsaved page index (already done in init_cache, just return the result)"""
        return self.first_unsaved_page

    def get_page_info(self, page_index):
        """Get annotation information for the specified page"""
        assert page_index >= 0 and page_index <= self.last_page_index, "Page index out of range"

        filekey, page_id = self.page_list[page_index]
        image_path = self.img_paths[filekey][page_id]

        # Read annotation information from the cache
        annotation = self.page_annotation_cache[filekey][page_id]
        
        page_info = f"Page {page_index + 1}/{len(self.page_list)}: {filekey} {page_id}"
        return image_path, annotation, page_info

    def save_annotation(self, page_index, page_ann_dict):
        """Update cache and save the final annotation results"""
        # Check if page_ann_dict is filled out
        if page_ann_dict['keep'] is None:
            return False, "Please select whether to keep this page"
        elif page_ann_dict['keep'] == 'yes':
            missing_fields = []
            for field_name in self.field_names:
                if field_name not in page_ann_dict or page_ann_dict[field_name] is None:
                    missing_fields.append(field_name)
            if missing_fields:
                return False, f"Please fill in {', '.join(missing_fields)}"
        
        # Check if page_index is out of range
        if page_index < 0 or page_index > self.last_page_index:
            return False, "Page index out of range"
        
        filekey, page_id = self.page_list[page_index]
        output_file = os.path.join(self.candidate_dir, filekey, self.save_ann_file)
 
        # Update cache
        self.page_annotation_cache[filekey][page_id] = page_ann_dict
        
        # Mark this page as saved
        self.page_save_flag[page_index] = True
        # Save only the validated pages for this filekey
        validated_pages = {}
        for pid, annotation in self.page_annotation_cache[filekey].items():
            if self.page_save_flag[self.page_idx[filekey][pid]]:
                validated_pages[pid] = annotation
        
        # Save file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(validated_pages, f, ensure_ascii=False, indent=2)
            return True, f'✅ Page {page_index + 1} saved successfully'
        except Exception as e:
            return False, f"❌ Failed to save page {page_index + 1}: {e}"
    
    def save_all_annotation(self):
        """
        Save all annotation information to the root directory, save kept page information to the selected directory,
        and save deleted page information to the trash directory.
        Ensure all pages are saved before calling.
        
        Returns:
            keep_num: Number of pages kept
        """
        all_annotation_save_file = os.path.join(self.base_dir, self.save_ann_file)
        with open(all_annotation_save_file, 'w', encoding='utf-8') as f:
            json.dump(self.page_annotation_cache, f, ensure_ascii=False, indent=2)
        keep_num = 0
        for filekey, page_annotation in self.page_annotation_cache.items():
            os.makedirs(os.path.join(self.selected_dir, filekey), exist_ok=True)
            os.makedirs(os.path.join(self.trash_dir, filekey), exist_ok=True)
            for page_id, page_annotation in page_annotation.items():
                encode_page_id = page_id.replace(':', '_').replace(';', '-')
                if page_annotation['keep'] == 'yes':
                    shutil.copy(os.path.join(self.candidate_dir, filekey, encode_page_id + '.png'), os.path.join(self.selected_dir, filekey, encode_page_id + '.png'))
                    keep_num += 1
                else:
                    shutil.copy(os.path.join(self.candidate_dir, filekey, encode_page_id + '.png'), os.path.join(self.trash_dir, filekey, encode_page_id + '.png'))
            # Remove empty directories
            if not os.listdir(os.path.join(self.selected_dir, filekey)):
                os.rmdir(os.path.join(self.selected_dir, filekey))
            if not os.listdir(os.path.join(self.trash_dir, filekey)):
                os.rmdir(os.path.join(self.trash_dir, filekey))
        report = {
            'keep_num': keep_num,
            'page_num': len(self.page_list),
            'rate': round(keep_num / len(self.page_list), 2),
        }
        with open(os.path.join(self.base_dir, 'report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return keep_num

def create_gradio_interface(base_dir=None, save_ann_file=None, start_page_index=-1):
    # Configuration parameters
    if base_dir is None or save_ann_file is None:
        raise ValueError("base_dir and save_ann_file are required")
    
    validator = AnnotationValidator(base_dir, save_ann_file)
    if start_page_index == -1:
        start_page_index = validator.earliest_unsaved_page
        print(f"Automatically detected earliest unsaved page: {start_page_index + 1}")
    else:
        print(f"Starting from specified page: {start_page_index + 1}")
    
    def load_page(page_index):
        """Load page data
        
        Args:
            page_index: The page index.

        Returns:
            image_path: The image path.
            page_info: The page information.
            save_ann_dict: The saved annotation data.
        """
        image_path, save_ann_dict, page_info = validator.get_page_info(page_index)
        return image_path, page_info, save_ann_dict

    def prev_page(current_index, save_ann_dict):
        """When information is complete, save automatically.
        If incomplete or on the first page, stay on the current page and give a prompt, otherwise switch to the previous page and load its data

        Args:
            current_index: The current page index.
            save_ann_dict: The saved annotation data.

        Returns:
            image_path: The image path.
            page_info: The page information.
            save_ann_dict: The saved annotation data.
            index: The switched page index.
            message: The message.
        """
        is_first_page = current_index == 0
        # Automatically save the current page
        success_flag, message = validator.save_annotation(current_index, save_ann_dict)
        if is_first_page:
            if success_flag:
                message = f"❗ This is the first page, {message}"
            new_index = current_index
        else:
            new_index = current_index - 1 if success_flag else current_index
        image_path, page_info, save_ann_dict = load_page(new_index)
        return image_path, page_info, save_ann_dict, new_index, message

    def next_page(current_index, save_ann_dict):
        """When information is complete, save automatically.
        If incomplete or on the last page, stay on the current page and give a prompt, otherwise switch to the next page and load its data.
        When on the last page, also save all annotation information

        Args:
            current_index: The current page index.
            save_ann_dict: The saved annotation data.

        Returns:
            image_path: The image path.
            page_info: The page information.
            save_ann_dict: The saved annotation data.
            index: The switched page index.
            message: The message.
        """
        is_last_page = current_index == validator.last_page_index
        # Automatically save the current page
        success_flag, message = validator.save_annotation(current_index, save_ann_dict)
        if is_last_page:
            if success_flag:
                keep_num = validator.save_all_annotation()
                message = f"❗ This is the last page, {message}, \n{keep_num} pages kept"
            new_index = current_index
        else:
            new_index = current_index + 1 if success_flag else current_index
        image_path, page_info, save_ann_dict = load_page(new_index)
        return image_path, page_info, save_ann_dict, new_index, message

    def handle_ann_field_change(save_ann_dict, field_name, value):
        """Handle annotation field changes"""
        save_ann_dict[field_name] = value
        return save_ann_dict
    
    def handle_page_change(save_ann_dict):
        """Handle page switching"""
        # Update UI components
        return (*(save_ann_dict.get(field_name) for field_name in validator.field_names), )  


    # Create Gradio interface
    with gr.Blocks(title="🎨 UI Annotation Validation Tool", theme=gr.themes.Soft()) as demo:
        # Add custom CSS
        gr.HTML("""
        <style>
        .gradio-container {
            max-width: 100% !important;
        }
        .status-complete {
            background-color: #d4edda !important;
            color: #155724 !important;
            border: 1px solid #c3e6cb !important;
            border-radius: 4px;
            padding: 8px;
        }
        .status-incomplete {
            background-color: #f8d7da !important;
            color: #721c24 !important;
            border: 1px solid #f5c6cb !important;
            border-radius: 4px;
            padding: 8px;
        }
        </style>
        """)
        
        # Layout
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=1):
                    image_display = gr.Image(
                        label="🖼️ Page Screenshot",
                        type="filepath",
                    )
                
                with gr.Column(scale=1):
                    with gr.Row():
                        page_info_box = gr.Textbox(
                            label="📄 Current Page Info", 
                            interactive=False,
                            value=""
                        )
                        message_box = gr.Textbox(
                            label="📊 Output Info",
                            interactive=False,
                            value=""
                        )
                    with gr.Row():
                        prev_btn = gr.Button("⬅️ Previous", variant="primary", size="sm")
                        next_btn = gr.Button("➡️ Next", variant="primary", size="sm")
                    with gr.Row():
                        # Keep or not
                        keep_radio_box = gr.Radio(
                            label="Keep",
                            choices=["yes", "no"],
                            value="yes",
                        )
                    
                    with gr.Group(visible=True) as annotation_group:
                        with gr.Column():
                            # Show other fields only when kept
                            radio_box_dict = {}
                            for field_name in validator.field_names:
                                if field_name == 'keep':
                                    continue
                                radio_box_dict[field_name] = gr.Radio(
                                    label=field_name,
                                    choices=validator.annotation_fields[field_name]['options'],
                                    value=None
                                )
        
        # State variables
        page_index = gr.State(0)
        save_ann_dict_state = gr.State({})
        
        # Initial loading
        demo.load(
            fn=lambda: load_page(start_page_index),
            outputs=[image_display, page_info_box, save_ann_dict_state]
        ).then(
            fn=lambda: start_page_index,
            outputs=[page_index]
        ).then(
            fn=handle_page_change,
            inputs=[save_ann_dict_state],
            outputs=[keep_radio_box, *radio_box_dict.values()]
        )

        # Navigation button event binding
        prev_btn.click(
            fn=prev_page,
            inputs=[page_index, save_ann_dict_state],
            outputs=[image_display, page_info_box, save_ann_dict_state, page_index, message_box]
        ).then(
            fn=handle_page_change,
            inputs=[save_ann_dict_state],
            outputs=[keep_radio_box, *radio_box_dict.values()]
        )
        
        next_btn.click(
            fn=next_page,
            inputs=[page_index, save_ann_dict_state],
            outputs=[image_display, page_info_box, save_ann_dict_state, page_index, message_box]
        ).then(
            fn=handle_page_change,
            inputs=[save_ann_dict_state],
            outputs=[keep_radio_box, *radio_box_dict.values()]
        )
        
        # Annotation field update event binding
        keep_radio_box.change(
            fn=lambda save_ann_dict, value : handle_ann_field_change(save_ann_dict, "keep", value),
            inputs=[save_ann_dict_state, keep_radio_box],
            outputs=[save_ann_dict_state]
        ).then(
            fn=lambda x: gr.update(visible=x == "yes"),
            inputs=[keep_radio_box],
            outputs=[annotation_group]
        )
        
        for field_name in validator.field_names:
            if field_name == 'keep':
                continue
            radio_box_dict[field_name].change(
                fn=lambda save_ann_dict, value, field=field_name : handle_ann_field_change(save_ann_dict, field, value),
                inputs=[save_ann_dict_state, radio_box_dict[field_name]],
                outputs=[save_ann_dict_state]
            )
    
    return demo

def find_free_port():
    """Find an available port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def main():
    parser = argparse.ArgumentParser(description='Launch UI Annotation Validation Tool')
    
    parser.add_argument(
        '--base_dir', 
        type=str, 
        default="./output/page_filter",
        help='Root directory path containing page directories'
    )

    parser.add_argument(
        '--save_ann_file', 
        type=str,
        default='final_annotation.json',
        help='Filename for saving annotations'
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=0,
        help='Server port number, set to 0 to automatically select an available port'
    )
    
    parser.add_argument(
        '--host', 
        type=str, 
        default="127.0.0.1",
        help='Server host address'
    )
    
    parser.add_argument(
        '--temp_dir', 
        type=str, 
        default='./gradio_temp',
        help='Custom temporary directory path (for solving permission issues)'
    )
    
    parser.add_argument(
        '--start_page', 
        type=int, 
        default=-1,
        help='Index of the first page to load, -1 means automatically start from the earliest unsaved page'
    )
    
    args = parser.parse_args()
    
    # If the port is set to 0, automatically select a free port
    if args.port == 0:
        args.port = find_free_port()
        print(f"Automatically selected available port: {args.port}")
    
    # If a custom temporary directory is specified, use it
    if args.temp_dir:
        os.environ['GRADIO_TEMP_DIR'] = os.path.abspath(args.temp_dir)
        os.environ['GRADIO_CACHE_DIR'] = os.path.join(os.path.abspath(args.temp_dir), 'cache')
        os.makedirs(os.environ['GRADIO_TEMP_DIR'], exist_ok=True)
        os.makedirs(os.environ['GRADIO_CACHE_DIR'], exist_ok=True)
        print(f"Using custom temporary directory: {args.temp_dir}")
    
    # Determine the starting page index
    start_page_index = args.start_page
    
    print(f"Starting UI Annotation Validation Tool...")
    print(f"Base directory: {args.base_dir}")
    print(f"Save annotation file: {args.save_ann_file}")
    print(f"Server address: http://{args.host}:{args.port}")
    
    # Create and start the application
    demo = create_gradio_interface(
        base_dir=args.base_dir,
        save_ann_file=args.save_ann_file,
        start_page_index=start_page_index
    )
    
    # Configure launch parameters
    launch_kwargs = {
        'server_name': args.host,
        'server_port': args.port,
        'share': False,
        'show_error': True
    }
    
    # Launch the application
    try:
        demo.launch(**launch_kwargs)
    except Exception as e:
        raise e

if __name__ == "__main__":
    try:
        from ...configs.paths import enter_project_root
        enter_project_root()
    except Exception as e:
        pass
    main()