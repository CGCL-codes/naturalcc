from utils.utils import *
import re
from utils.log import logger
import torch

SEED = 2026
BACKBONE=None # f(prompt, text, [imgs], temperature, seed, n, tensor_parallel_size)

class MLLMAgent:
    def __init__(self, prompt):
        self.prompt = prompt
        self.do_sample = False

    def infer(self, textx_imgs, parse=True, temperature=0, seed=SEED, 
                 n=1):
        self.do_sample = (n>1)
        text = BACKBONE(self.prompt, textx_imgs, temperature, seed, n)
        return self.parser(text) if parse else text
    
    def parser(self, text):
        return text


class AgentGenerate(MLLMAgent):
    def __init__(self):
        super().__init__("""
You are an expert Tailwind developer.

Based on the reference screenshot of a specific section of a webpage (such as the header, footer, card, etc.) provided by the user, build a single-page app using Tailwind, HTML, and JS. Please follow the detailed requirements below to ensure the generated code is accurate:

### Basic Requirements:
                         
1. **Rigid Requirements**
   - You are provided with the following unmodifiable HTML framework:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
</head>
<body>
    <!-- Your task is to fill this area -->
</body>
</html>
```
   - Your task is to generate a code block that starts with a <div> tag and ends with a </div> tag, and embed it within the <body> tag of the above - mentioned framework.
   - Do not deliberately center the content. Arrange the elements according to their original layout and positions.
   - The generated code should not have fixed width and height settings.
   - Ensure that the proportions of images in the code are preserved.
   - Both the margin and padding in the code should be set to 0.
   - Make sure that the generated code does not conflict with outer <div> elements in terms of layout and style.
   - The final return should be the complete HTML code, that is, including the above - mentioned framework and the code you generated and embedded into the <body> of the framework.

2. **Appearance and Layout Consistency:**
   - Ensure the app looks exactly like the screenshot, including the position, hierarchy, and content of all elements.
   - The generated HTML elements and Tailwind classes should match those in the screenshot, ensuring that text, colors, fonts, padding, margins, borders, and other styles are perfectly aligned.

3. **Content Consistency:**
   - Use the exact text from the screenshot, ensuring the content of every element matches the image.
   - For images, use placeholder images from https://placehold.co and include a detailed description in the alt text for AI-generated images.

4. **No Comments or Placeholders:**
   - Do not add comments like "<!-- Add other navigation links as needed -->" or "<!-- ... other news items ... -->". Write the full, complete code for each element.

5. **Libraries to Use:**
   - Use the following libraries:
     - Google Fonts: Use the relevant fonts from the screenshot.
                         

### Process Steps:

1. **Analyze the Section:**
   Based on the provided screenshot, analyze a specific section of the webpage (such as the header, footer, card, form, etc.). Break down all the elements in this section (e.g., text, images, buttons, etc.) and understand their relative positions and hierarchy.

2. **Generate HTML Code:**
   Based on the analysis from Step 1, generate a complete HTML code snippet representing that specific section, ensuring all elements, positions, and styles match the screenshot.

3. **Text Content Comparison:**
   Compare the generated HTML with the screenshot’s text content to ensure accuracy. If there are any discrepancies or missing content, make corrections.

4. **Color Comparison:**
   Compare the text color and background color in the generated HTML with those in the screenshot. If they don't match, adjust the Tailwind classes and styles to reflect the correct colors.

5. **Background and Other Style Comparison:**
   Ensure the background colors, borders, padding, margins, and other styles in the generated HTML accurately reflect the design shown in the screenshot.

6. **Final Integration:**
   After reviewing and refining the previous steps, ensure that the generated HTML code is complete and perfectly matches the specific section of the screenshot.

### Code Format:

Please return the complete HTML code 
                         """)

    def parser(self, text):
        patern = r'```html(.*?)```' #r'```html(.*?)(?:```|$)' 
        if self.do_sample:
          code_blocks=[re.findall(patern, t, re.DOTALL)[0] for t in text]
        else:
          code_blocks = re.findall(patern, text, re.DOTALL)[0]
        return code_blocks if code_blocks else text  
      

class AgentGenerateElf(MLLMAgent): # simplified generator
    def __init__(self):
        super().__init__("""
You are an expert Tailwind developer.

Based on the reference screenshot of a specific section of a webpage (such as the header, footer, card, etc.) provided by the user, build a single-page app using Tailwind, HTML, and JS. 
Please follow the detailed requirements below to ensure the generated code is accurate:

### Basic Requirements:
                         
1. **Rigid Requirements**
   - You are provided with the following unmodifiable HTML framework.:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
</head>
<body>
    <!-- Your task is to fill this area -->
</body>
</html>
```
2. **Appearance and Layout Consistency:**
  
3. **Content Consistency:**
   - Use the exact text from the screenshot, ensuring the content of every element matches the image.
   - For images, use placeholder images from https://placehold.co and include a detailed description in the alt text for AI-generated images.

4. **No Comments **

5. **Libraries to Use:**
   - Use the following libraries:
     - Google Fonts: Use the relevant fonts from the screenshot.

### Code Format:

Please return the complete HTML code without any explanations or justifications.
                         """)

    def parser(self, text):
        patern = r'```html(.*?)```' #r'```html(.*?)(?:```|$)' 
        if self.do_sample:
          codes = []
          for t in text:
            try:
               code = re.findall(patern, t, re.DOTALL)[0]
               codes.append(code)
            except:
              continue
          return codes
        else:
          codes = re.findall(patern, text, re.DOTALL)[0]
        return codes if codes else text  

# class AgentGenerate(MLLMAgent):
#     def __init__(self):
#         super().__init__("""
# You are an expert Tailwind developer.

# Based on the reference screenshot of a specific section of a webpage (such as the header, footer, card, etc.) provided by the user, build a single-page app using Tailwind, HTML, and JS. Please follow the detailed requirements below to ensure the generated code is accurate:

# ### Basic Requirements:
                         
# 1. **Rigid Requirements**
#    - You are provided with the following unmodifiable HTML framework:
# ```html
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <script src="https://cdn.tailwindcss.com"></script>
#     <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
# </head>
# <body>
#     <!-- Your task is to fill this area -->
# </body>
# </html>
# ```
#    - Your task is to generate a code block that starts with a <div> tag and ends with a </div> tag, and embed it within the <body> tag of the above - mentioned framework.
#    - The generated code should not have fixed width and height settings.
#    - Ensure that the proportions of images in the code are preserved.
#    - Both the margin and padding in the code should be set to 0.
#    - Make sure that the generated code does not conflict with outer <div> elements in terms of layout and style.
#    - The final return should be the complete HTML code, that is, including the above - mentioned framework and the code you generated and embedded into the <body> of the framework.

# 2. **Appearance and Layout Consistency:**
#    - Ensure the app looks exactly like the screenshot, including the position, hierarchy, and content of all elements.
#    - The generated HTML elements and Tailwind classes should match those in the screenshot, ensuring that text, colors, fonts, padding, margins, borders, and other styles are perfectly aligned.

# 3. **Content Consistency:**
#    - Use the exact text from the screenshot, ensuring the content of every element matches the image.
#    - For images, use placeholder images from https://placehold.co and include a detailed description in the alt text for AI-generated images.

# 4. **No Comments or Placeholders:**
#    - Do not add comments like "<!-- Add other navigation links as needed -->" or "<!-- ... other news items ... -->". Write the full, complete code for each element.

# 5. **Libraries to Use:**
#    - Use the following libraries:
#      - Tailwind CSS: <script src="https://cdn.tailwindcss.com"></script>
#      - Google Fonts: Use the relevant fonts from the screenshot.
#      - Font Awesome icons: <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>
                         

# ### Process Steps:

# 1. **Analyze the Section:**
#    Based on the provided screenshot, analyze a specific section of the webpage (such as the header, footer, card, form, etc.). Break down all the elements in this section (e.g., text, images, buttons, etc.) and understand their relative positions and hierarchy.

# 2. **Generate HTML Code:**
#    Based on the analysis from Step 1, generate a complete HTML code snippet representing that specific section, ensuring all elements, positions, and styles match the screenshot.

# 3. **Text Content Comparison:**
#    Compare the generated HTML with the screenshot’s text content to ensure accuracy. If there are any discrepancies or missing content, make corrections.

# 4. **Color Comparison:**
#    Compare the text color and background color in the generated HTML with those in the screenshot. If they don't match, adjust the Tailwind classes and styles to reflect the correct colors.

# 5. **Background and Other Style Comparison:**
#    Ensure the background colors, borders, padding, margins, and other styles in the generated HTML accurately reflect the design shown in the screenshot.

# 6. **Final Integration:**
#    After reviewing and refining the previous steps, ensure that the generated HTML code is complete and perfectly matches the specific section of the screenshot.

# ### Code Format:

# Please return the complete HTML code for this specific section, ensuring all elements, styles, and text are correctly implemented as shown in the screenshot. The generated code should be enclosed within `<html></html>` tags and ready for use in the webpage.
# """)

#     def parser(self, text):
#         code_blocks = re.findall(r'```html(.*?)```', text, re.DOTALL)
#         return code_blocks[0] if code_blocks else text    
        


class AgentAssemble(MLLMAgent):
    def __init__(self):
        super().__init__("""
        You are an experienced front-end developer tasked with assembling multiple webpage module codes into a complete webpage.

        # CONTEXT #
        I will provide a screenshot of a webpage, the location information for each module, and the corresponding module code. 
        Your task is to assemble these modules into a complete webpage code based on their positions.

        # OBJECTIVE #
        Generate a complete HTML file that ensures the layout, style, and content of each module match the original webpage.

        # RESPONSE #
        You need to return the final assembled complete HTML code, for example:
        ```html
        code
        ```

        # steps #
        Please follow the steps below:

        **step1**: Analyze the webpage screenshot and the position information of each module.
        - Based on the screenshot and the module position data, understand the relative placement and layout of each module.
        - The position of each module is defined by two coordinates: the top-left corner [x1, y1] and the bottom-right corner [x2, y2]. x1 < x2 and y1 < y2. The coordinate values range from 0 to 1, representing the ratio relative to the width and height of the image.

        **step2**: Assemble the HTML code of each module based on its position.
        - Use the provided module code to stitch the modules together in the correct order and position.
        - Ensure that the modules do not overlap and that the layout is correct.

        **step3**: Review and fix the assembled webpage.
        - Compare the generated webpage with the screenshot to ensure the content, layout, and styles match exactly.
        - If any issues such as misalignment, overlapping, or missing content are found, fix them.

        **step4**: Generate the final HTML code.
        - Based on the checks and fixes from step 3, generate the final complete webpage code that closely matches the screenshot.

        # Notes #
        - There should be no overlap between modules.
        - For image modules, use placeholder images from https://placehold.co and provide a detailed description in the alt text for AI-based image generation.
        - Pay attention to details such as background color, text color, font size, padding, margins, borders, and other visual elements.
        - **Do not omit any module's code**. Every module, regardless of its size or complexity, must be included in the final HTML code. Ensure that each module's functionality and layout are represented fully in the assembled page.
                         
        # Libraries #
        - You may use Google Fonts to match the fonts in the screenshot.
        - Use the Font Awesome icon library: <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>
        """)
    def parser(self, text):
        match = re.search(r'```html(.*?)```', text, re.DOTALL)

        if match:
            return match.group(1)  
        else:
            return None  



class AgentEvaluate4image(MLLMAgent):
    def __init__(self):
        super().__init__("""
You are an experienced front-end developer tasked with evaluating three rendered images (Images 2, 3, and 4) based on the original webpage design (Image 1). For each image, you will provide scores across four categories, comparing them to the reference design.

### Comparison Framework #

#### Step 1: Structural and Layout Analysis (Highest Priority) - 40 points
1. **Overall Layout**:
   - Compare the structure of the entire page, identifying missing, extra, or misplaced sections:
     - Example: "The sidebar is present in the reference but missing in the output."
     - Example: "The footer is rendered at the top of the page instead of the bottom."

2. **Alignment and Positioning**:
   - Evaluate the alignment and positioning of all elements:
     - Example: "The search bar is centered in the reference but misaligned to the right in the output."

3. **Spacing and Proportions**:
   - Compare spacing and proportions between elements:
     - Example: "The gap between the header and main content is 20px in the reference but only 5px in the output."
     - Example: "The button dimensions are 100x40px in the reference but 90x50px in the output."

4. **Missing or Extra Elements**:
   - List any elements that are missing or unexpectedly present:
     - Example: "The output includes an extra icon next to the navigation links, which is not present in the reference."

#### Step 2: Styling and Color Analysis (High Priority) - 30 points
1. **Color Scheme Consistency**:
   - Compare all colors (e.g., backgrounds, borders, text) across the design:
     - Example: "The primary button background is blue (#0000FF) in the reference but green (#00FF00) in the output."
     - Example: "Gradient effects are missing in the output, appearing as flat colors."

2. **Borders, Shadows, and Effects**:
   - Assess consistency in borders, shadows, opacity, and visual effects for all elements:
     - Example: "The header has a subtle shadow in the reference, but it is absent in the output."
     - Example: "The button border is rounded in the reference but appears square in the output."

#### Step 3: Text and Typography Analysis (Medium Priority) - 20 points
1. **Font and Styling**:
   - Verify font type, size, weight, and alignment for all text:
     - Example: "The header text uses a bold font (700 weight) in the reference but is rendered as regular (400 weight) in the output."

2. **Text Placement and Content**:
   - Ensure the placement and content of all text is consistent with the reference:
     - Example: "The 'Contact Us' heading appears above the form in the reference but below it in the output."

#### Step 4: Placeholder Content and Imagery (Low Priority) - 10 points
1. **Placeholder Consistency**:
   - Compare the dimensions, alignment, and proportions of all placeholder content:
     - Example: "The placeholder image dimensions are 200x200px in the reference but 180x180px in the output."

2. **Filler Text**:
   - Ensure that placeholder text matches the reference in size, alignment, and placement:
     - Example: "The lorem ipsum text is left-aligned in the reference but appears centered in the output."

### Initialization #
I will provide four images:
1. **Original Webpage Design** (Reference Image).
2. Three **Rendered Images** generated from the predicted HTML code.

You will evaluate Images 2, 3, and 4 by comparing them with the reference image (Image 1) based on the criteria above. For each image, provide four scores representing the evaluation in the following order: **Structural and Layout Analysis**, **Styling and Color Analysis**, **Text and Typography Analysis**, **Placeholder Content and Imagery**.

### Output Format #
Your output should consist of three lines, each with four numbers (representing the scores for each criterion). For example:

40, 30, 20, 10  
30, 20, 20, 10  
40, 10, 10, 10

Please provide only the scores, without any explanations or justifications.
""")
    def parser(self, text):
        lines = text.strip().split('\n')
        numbers = []
        for line in lines:
            row_numbers = list(map(int, line.split(',')))
            row_numbers.extend([sum(row_numbers)])
            numbers.append(row_numbers)
        return numbers
     

# class Evaluator(MLLMAgent):
#     def __init__(self):
#         super().__init__("""
# {
#   "task": {
#     "description": "You are an experienced frontend engineer tasked with evaluating the quality of generated web pages. 
#     Your evaluation will compare screenshots of generated web pages against a reference design screenshot, provided as the first image. 
#     Using the provided evaluation_criteria and following the evaluation_process and key_notes, you will score each generated image and provide output strictly in the specified output_format.",
#     "input": {
#       "reference_image": "The first image is the Reference Design.",
#       "generated_images": "The remaining images are screenshots of the Renderd Generated Web Pages."
#     },
#     "evaluation_criteria": [
#       {
#         "name": "Structural and Layout Analysis",
#         "priority": "Highest",
#         "score_range": "0-40",
#         "sub_criteria": {
#           "overall_layout": "Identify structural differences such as missing, extra, or misplaced sections.",
#           "alignment_and_positioning": "Check element alignment and relative positioning.",
#           "spacing_and_proportions": "Compare spacing and size proportions between elements.",
#           "missing_or_extra_elements": "List discrepancies in element presence."
#         }
#       },
#       {
#         "name": "Styling and Color Analysis",
#         "priority": "High",
#         "score_range": "0-30",
#         "sub_criteria": {
#           "color_scheme_consistency": "Assess color differences in backgrounds, borders, or text.",
#           "borders_shadows_effects": "Evaluate consistency in visual effects like borders, shadows, and opacity."
#         }
#       },
#       {
#         "name": "Text and Typography Analysis",
#         "priority": "Medium",
#         "score_range": "0-20",
#         "sub_criteria": {
#           "font_and_styling": "Check font type, size, weight, and alignment.",
#           "text_placement_and_content": "Verify text placement and content accuracy."
#         }
#       },
#       {
#         "name": "Placeholder Content and Imagery",
#         "priority": "Low",
#         "score_range": "0-10",
#         "sub_criteria": {
#           "placeholder_consistency": "Compare dimensions, alignment, and proportions of placeholders.",
#           "filler_text": "Verify alignment, placement, and consistency of placeholder text."
#         }
#       }
#     ],
#     "evaluation_process": {
#       "requirement": "For each evaluation criterion, consider all the generated web pages together before assigning scores. Compare the performance of all pages for that specific criterion, then score each page relative to the reference design and the other pages in the batch.",
#       "steps": [
#         "Step 1: For a given criterion (e.g., Structural and Layout Analysis), review all generated images simultaneously, then assign scores for each generated image based on their relative accuracy and compliance with the criterion.",
#         "Step 2: Ensure that scores are consistent, with the best-performing image receiving the highest score and others graded proportionally."
#       ],
#       "note": "This process ensures fairness and consistency in scoring by incorporating a horizontal comparison of all generated pages."
#     },
#     "output_format": {
#       "description": "Provide similarity scores for each generated screenshot in one line per image, separated by commas.",
#       "examples": [
#         {
#           "case": "2 images total (1 reference + 1 generated)",
#           "output": "11, 22, 3, 6"
#         },
#         {
#           "case": "4 images total (1 reference + 3 generated)",
#           "output": 
#             "11, 2, 7, 9
#             3, 25, 15, 8
#             35, 28, 18, 9"
#           ]
#         }
#       ],
#       "note": "Do not include explanations or justifications in the output."
#     },
#     "key_notes": [
#       "Evaluate all criteria simultaneously for all generated screenshots within a batch.",
#       "Provide scores as integers within the specified range for each criterion.",
#       "Ensure fairness by comparing all pages horizontally and assigning relative scores for each criterion.",
#       "Strictly adhere to the output_format; provide scores in the required format without additional explanations."
#     ]
#   }
# }

# """)
#     def infer(self, imgs=[], parse = True):
#       user_text =f"""
#       For this case, {len(imgs)} images are provided: the first is the reference, and the next {len(imgs)-1} are the screenshots of generated webpages to be assessed. 
#       For each generated webpage, follow the evaluation criteria described in the prompt and provide {len(imgs)-1} lines of scores. 
#       Please pay attention to the relative comparison across all images when giving scores.
#       """
#       text = BACKBONE(self.prompt, user_text, imgs)
#       return self.parser(text) if parse else text
      
#     def parser(self, text:str):
#       lines = text.strip().strip('```').strip('\n').split('\n')
#       numbers = []
#       for line in lines:
#         row_numbers = list(map(int, line.split(',')))
#         row_numbers.extend([sum(row_numbers)])
#         numbers.append(row_numbers)
#       return numbers


class Evaluator(MLLMAgent):
    def __init__(self):
        super().__init__("""
{
  "task": {
    "description": "You are an experienced frontend engineer tasked with evaluating the quality of generated web pages. 
    Your evaluation will compare screenshots of generated web pages against a reference design screenshot, provided as the first image. 
    Using the provided evaluation_criteria and following the evaluation_process and key_notes, you will score each generated image and provide output strictly in the specified output_format.",
    "input": {
      "reference_image": "The first image is the Reference Design.",
      "generated_images": "The remaining images are screenshots of the Rendered Generated Web Pages."
    },
    "evaluation_criteria": [
      {
        "name": "Structural and Layout Analysis",
        "priority": "Highest",
        "score_range": "0-40",
        "sub_criteria": {
          "overall_layout": "Identify structural differences such as incorrect global structure (e.g., top-to-bottom vs. left-to-right layout), missing, extra, or misplaced sections.",
          "module_placement": "Verify that each module (e.g., header, footer, cards, sidebars) appears in the correct position relative to other modules.",
          "missing_or_extra_elements": "List discrepancies in element presence, including overlooked modules or incorrectly added components."
        }
      },
      {
        "name": "Styling and Color Analysis",
        "priority": "Medium",
        "score_range": "0-25",
        "sub_criteria": {
          "color_scheme_consistency": "Assess color differences in backgrounds, borders, or text, ensuring consistency with the reference design.",
          "borders_shadows_effects": "Evaluate consistency in visual effects like borders, shadows, and opacity, including adherence to subtle design details."
        }
      },
      {
        "name": "Text and Typography Analysis",
        "priority": "Medium",
        "score_range": "0-25",
        "sub_criteria": {
          "text_placement_and_content": "Verify text placement, alignment, and content accuracy, ensuring no truncation, misalignment, or text overflow.",
          "font_and_styling": "Check font type, size, weight, and alignment against the reference design."
        }
      },
      {
        "name": "Placeholder Content and Imagery",
        "priority": "Low",
        "score_range": "0-10",
        "sub_criteria": {
          "placeholder_consistency": "Compare dimensions, alignment, and proportions of placeholders, ensuring consistency with the reference layout.",
          "filler_text": "Verify alignment, placement, and consistency of placeholder text, ensuring no visual distortion or errors."
        }
      }
    ],
    "evaluation_process": {
      "requirement": "For each evaluation criterion, consider all the generated web pages together before assigning scores. Compare the performance of all pages for that specific criterion, then score each page relative to the reference design and the other pages in the batch.",
      "steps": [
        "Step 1: For a given criterion (e.g., Structural and Layout Analysis), review all generated images simultaneously, then assign scores for each generated image based on their relative accuracy and compliance with the criterion.",
        "Step 2: Ensure that scores are consistent, with the best-performing image receiving the highest score and others graded proportionally."
      ],
      "note": "This process ensures fairness and consistency in scoring by incorporating a horizontal comparison of all generated pages."
    },
    "output_format": {
      "description": "Provide similarity scores for each generated screenshot in one line per image, separated by commas.",
      "examples": [
        {
          "case": "2 images total (1 reference + 1 generated)",
          "output": "11, 22, 3, 6"
        },
        {
          "case": "4 images total (1 reference + 3 generated)",
          "output": 
            "11, 2, 7, 9
            3, 23, 15, 8
            35, 22, 18, 9"
          ]
        }
      ],
      "note": "Do not include explanations or justifications in the output."
    },
    "key_notes": [
      "Evaluate all criteria simultaneously for all generated screenshots within a batch.",
      "For Structural and Layout Analysis, ensure strict adherence to the reference layout structure (e.g., top-to-bottom, left-to-right). Pay particular attention to module placement and global layout correctness.",
      "Provide scores as integers within the specified range for each criterion.",
      "Ensure fairness by comparing all pages horizontally and assigning relative scores for each criterion.",
      "Strictly adhere to the output_format; provide scores in the required format without additional explanations."
    ]
  }
}

""")
    def infer(self, imgs=[], parse = True):
      user_text =f"""
      For this case, {len(imgs)} images are provided: the first is the reference, and the next {len(imgs)-1} are the screenshots of generated webpages to be assessed. 
      For each generated webpage, follow the evaluation criteria described in the prompt and provide {len(imgs)-1} lines of scores. 
      Please pay attention to the relative comparison across all images when giving scores.
      """
      text = BACKBONE(self.prompt, [user_text] + imgs)
      return self.parser(text) if parse else text
      
    def parser(self, text:str):
      lines = text.strip().strip('```').strip('\n').split('\n')
      numbers = []
      for line in lines:
        row_numbers = list(map(int, line.split(',')))
        row_numbers.extend([sum(row_numbers)])
        numbers.append(row_numbers)
      return numbers

class AgentEvaluate2image(MLLMAgent):
    def __init__(self):
        super().__init__("""
You are an experienced front-end developer tasked with evaluating a rendered webpage (Image 2) based on the original webpage design (Image 1). For the rendered image, you will provide scores across four categories, comparing it to the reference design.

### Comparison Framework #

#### Step 1: Structural and Layout Analysis (Highest Priority) - 40 points
1. **Overall Layout**:
   - Compare the structure of the entire page, noting missing, extra, or misplaced sections:
     - Example: "The sidebar is present in the reference but missing in the output."
     - Example: "The footer is rendered at the top of the page instead of the bottom."

2. **Alignment and Positioning**:
   - Assess the alignment and positioning of elements:
     - Example: "The search bar is centered in the reference but misaligned to the right in the output."

3. **Spacing and Proportions**:
   - Compare the spacing and proportions between elements:
     - Example: "The gap between the header and main content is 20px in the reference but only 5px in the output."
     - Example: "The button dimensions are 100x40px in the reference but 90x50px in the output."

4. **Missing or Extra Elements**:
   - Identify any elements that are missing or unexpectedly present:
     - Example: "The output includes an extra icon next to the navigation links, which is not present in the reference."

#### Step 2: Styling and Color Analysis (High Priority) - 30 points
1. **Color Scheme Consistency**:
   - Compare the colors (backgrounds, borders, text) across the design:
     - Example: "The primary button background is blue (#0000FF) in the reference but green (#00FF00) in the output."
     - Example: "Gradient effects are missing in the output, appearing as flat colors."

2. **Borders, Shadows, and Effects**:
   - Assess consistency in borders, shadows, opacity, and visual effects for all elements:
     - Example: "The header has a subtle shadow in the reference, but it is absent in the output."
     - Example: "The button border is rounded in the reference but appears square in the output."

#### Step 3: Text and Typography Analysis (Medium Priority) - 20 points
1. **Font and Styling**:
   - Verify font type, size, weight, and alignment for all text:
     - Example: "The header text uses a bold font (700 weight) in the reference but is rendered as regular (400 weight) in the output."

2. **Text Placement and Content**:
   - Ensure the placement and content of all text matches the reference:
     - Example: "The 'Contact Us' heading appears above the form in the reference but below it in the output."

#### Step 4: Placeholder Content and Imagery (Low Priority) - 10 points
1. **Placeholder Consistency**:
   - Compare the dimensions, alignment, and proportions of all placeholder content:
     - Example: "The placeholder image dimensions are 200x200px in the reference but 180x180px in the output."

2. **Filler Text**:
   - Ensure that placeholder text matches the reference in size, alignment, and placement:
     - Example: "The lorem ipsum text is left-aligned in the reference but appears centered in the output."


### Initialize #
You will evaluate the rendered webpage (Image 2) by comparing it with the reference webpage (Image 1) based on the criteria above. For your evaluation, provide four scores in the following order: **Structural and Layout Analysis**, **Styling and Color Analysis**, **Text and Typography Analysis**, **Placeholder Content and Imagery**.

### Output Format #
Your response should be a single line containing a comma-separated list of four scores, corresponding to the four evaluation criteria. Please provide only the scores without explanations or justifications.

Example:
30, 20, 20, 10
""")
    def parser(self, text):
        lines = text.strip().split('\n')
        numbers = []
        for line in lines:
            row_numbers = list(map(int, line.split(',')))
            row_numbers.extend([sum(row_numbers)])
            numbers.append(row_numbers)
        return numbers[0]




# class AgentRefine(MLLMAgent):
#     def __init__(self):
#         super().__init__("""
# You are an experienced front-end developer, and your task is to refine the code based on expert ratings.

# ### Initialize #
# You will receive two images, one code, and a rating. The first image is the reference webpage (original design), and the second image is the rendered result of the provided code. You must modify the code based on the expert's ratings.

# The rating will consist of five numbers: the first four correspond to the four evaluation criteria, and the fifth is the overall score.

# ### Evaluation Criteria #

# #### Step 1: Structural and Layout Analysis (Highest Priority) - 40 points
# 1. **Overall Layout**:
#    - Compare the structure of the entire page, identifying missing, extra, or misplaced sections.
#      - Example: "The sidebar is present in the reference but missing in the output."
#      - Example: "The footer is rendered at the top instead of at the bottom."

# 2. **Alignment and Positioning**:
#    - Check the alignment and positioning of elements.
#      - Example: "The search bar is centered in the reference but misaligned to the right in the output."

# 3. **Spacing and Proportions**:
#    - Compare spacing and proportions between elements.
#      - Example: "The gap between the header and main content is 20px in the reference but 5px in the output."
#      - Example: "The button dimensions are 100x40px in the reference but 90x50px in the output."

# 4. **Missing or Extra Elements**:
#    - Identify any missing or extra elements.
#      - Example: "The output includes an extra icon next to the navigation links, which is not in the reference."

# #### Step 2: Styling and Color Analysis (High Priority) - 30 points
# 1. **Color Scheme Consistency**:
#    - Compare colors across the design (backgrounds, borders, text).
#      - Example: "The primary button background is blue (#0000FF) in the reference but green (#00FF00) in the output."
#      - Example: "Gradient effects are missing in the output, appearing as flat colors."

# 2. **Borders, Shadows, and Effects**:
#    - Evaluate the consistency of borders, shadows, opacity, and visual effects.
#      - Example: "The header has a subtle shadow in the reference but is absent in the output."
#      - Example: "The button border is rounded in the reference but appears square in the output."

# #### Step 3: Text and Typography Analysis (Medium Priority) - 20 points
# 1. **Font and Styling**:
#    - Verify font type, size, weight, and alignment for all text.
#      - Example: "The header text uses a bold font (700 weight) in the reference but is regular (400 weight) in the output."

# 2. **Text Placement and Content**:
#    - Ensure text placement and content match the reference.
#      - Example: "The 'Contact Us' heading appears above the form in the reference but below it in the output."

# #### Step 4: Placeholder Content and Imagery (Low Priority) - 10 points
# 1. **Placeholder Consistency**:
#    - Compare the dimensions, alignment, and proportions of placeholder content.
#      - Example: "The placeholder image dimensions are 200x200px in the reference but 180x180px in the output."

# 2. **Filler Text**:
#    - Ensure placeholder text matches the reference in size, alignment, and placement.
#      - Example: "The lorem ipsum text is left-aligned in the reference but appears centered in the output."

# ### Attention #
# If you believe the code does not need modification, return the original code without any changes.

# ### Response #
# Return the updated code in the following format:

# ```html
# [Updated HTML code]
# """)
#     def parser(self, text):
#       
#         match = re.search(r'```html(.*?)```', text, re.DOTALL)

#         if match:
#             return match.group(1) 
#         else:
#             return None  
        

class AgentGetText(MLLMAgent):
    def __init__(self):
        super().__init__("""
You are a senior expert in image recognition, specializing in parsing web page screenshots. 
Based on the web page screenshot I provide, please conduct a detailed analysis as required 
below and output relevant information:
    Text Extraction: Carefully identify the text information in the web page screenshot and clearly list the specific content within each text block. (Do not omit the description of any text just because it has a small number of words.)
    Text Location Positioning: For each part of the text information extracted, describe its approximate position in the web page screenshot. The position description should be as precise as possible, such as being located in the upper - left corner of the web page, or in the lower - middle part of the right side, etc.
    Text Relative Position Relationship: Clearly explain the relative position relationships between each piece of text information. For example, indicate that a certain text block is directly above, in the lower - left corner of, or immediately to the right of another text block.
Please ensure that the analysis is comprehensive and accurate, and the description is clear and easy to understand, to facilitate subsequent web - related development or optimization work.
                  
""")
        
class AgentRefine(MLLMAgent):
    def __init__(self):
        super().__init__("""
You are a professional web developer proficient in HTML and CSS. Your task is to optimize the given HTML code based on the information I provide, correcting minor flaws while retaining its original advantages. This HTML code was generated by splitting the web page into multiple modules for separate programming and then combining the code of each module. Although the overall layout is good, there may be issues such as missing elements, module boundaries, or incorrect relative positions of elements.

### Provided Information
1. **Two images**: The first is the original web page image, and the second is the rendering of the given code. The rendered image helps you understand the actual effect of the code.
2. **One piece of code**: That is, the HTML code to be optimized.
3. **One piece of web page text information**: Containing the accurate text content that should be on the web page.

### Tasks
1. **Text content check and correction**: Based on the provided web page text information, carefully check whether there are any omissions, redundancies, or errors in the text information in the code. If there are such issues, corrections are needed to ensure that the text content in the code is completely consistent with the given text information.
2. **Multi - dimensional analysis and adjustment**: By comparing the original web page image and the code rendering, analyze the problems in the existing code from the following dimensions and make necessary fine - tuning to meet the requirements of each dimension:
    - **Structure and layout**: Ensure that the page structure is complete, without missing, redundant, or mispositioned parts. At the same time, ensure that the alignment of elements is correct, text blocks are in the correct positions on the web page, and the spacing and proportions conform to the design of the original web page.
    - **Style and color**: Make all color settings such as background color, border color, and text color consistent with those of the original web page.
    - **Text and typography**: Ensure that the font type, size, weight, and alignment of the text match those of the original web page.
    - **Placeholder content and images**: Ensure that the size, alignment, and position of the placeholder content and images match those of the original web page.

### Notes
1. Only make necessary modifications and be sure not to damage the overall layout of the code. (The first criterion)
2. Pay attention to the proportion issue. Ensure that the length and width of the final code rendering are the same as those of the original web page. If there are blank spaces in the original web page, they must also exist in the code.
3. If images are involved, uniformly use placeholder images from https://placehold.co and add detailed and accurate image descriptions in the alt attribute text.

### Return Format
Please return the optimized code in the following format without adding additional information or explanations.

```html
(The complete optimized HTML code)
```
""")
    def parser(self, text):
        match = re.search(r'```html(.*?)```', text, re.DOTALL)

        if match:
            return match.group(1)  
        else:
            return None  