from .utils.gpt4o import gpt4o
import re

class Agent:
    def __init__(self, prompt):
        self.prompt = prompt

    def infer(self, image, text='', parse=True):
        text = gpt4o(self.prompt, image, text)
        return self.parser(text) if parse else text
    
    def parser(self, text):
        return text
    
class AgentSplit(Agent):
    def __init__(self):
        super().__init__("""你是一个擅长于搭建网页的网页工程师。
# CONTEXT #
我想实现一个将网页实现图片转换为实现该网页效果代码的项目。目前交给你的工作是将输入的网页图片分割为若干模块(例如：导航栏，菜单栏，内容模块，广告模块等)，以便于为后续针对各个模块进行更细致的开发和网页代码的编写。
# OBJECTIVE
将输入的网页图片分割为若干模块，以便于为后续针对各个模块进行更细致的开发和网页代码的编写。
# RESPONSE #
给出分割后的各模块的图片的坐标，图片左上角为原点，图片长宽分别记作1，给出模块左上角坐标(x1,y1)和右下角坐标(x2,y2)，最后输出模块名和(x1,y1,x2,y2), 保证x1<x2, y1<y2
# Initialize #
接下来的消息我会给你发送网页图片，收到后请按照以上规则输出分割后的各模块的名称以及坐标""")

    def parser(self, text):
        pattern = r"(\d+)\.\s+\*\*(.*?)\*\*\n\s+.*\((.*?)\)"
        matches = re.findall(pattern, text)

        # 解析坐标并存储为数组
        modules = []
        for match in matches:
            index, name, coords = match
            coords_list = [float(coord) for coord in coords.split(', ')]
            modules.append({'name': name, 'bbox': coords_list})

        return modules

class AgentI2C(Agent):
    def __init__(self):
        super().__init__("""你是一个擅长于搭建网页的网页工程师。
# CONTEXT #
我想实现一个将网页实现图片转换为实现该网页效果代码的项目。目前交给你的工作是根据分割后的网页模块的名称和图片，生成对应的HTML代码。
# OBJECTIVE
根据输入的网页图片和初始节点类型，生成局部的HTML代码。
# RESPONSE #
给出能够实现模块功能的局部HTML代码,附带有行内css。
# Initialize #
接下来的消息我会给你发送网页图片和模块名称，收到后请按照以上规则给出HTML代码（返回的HTML树最大层节点应当是给定的初始节点类型）""")
        
class AgentI2C2(Agent):
    def __init__(self):
        super().__init__("""You are a skilled web engineer with expertise in converting webpage images into fully functional HTML code.
CONTEXT
I am working on a project that converts webpage screenshots or design prototypes into corresponding web code. Your task is to generate detailed HTML and CSS based on the input webpage image, the given HTML DOM tree, and bbox information, ensuring that the generated webpage closely aligns with the original in both structure and appearance.
OBJECTIVE
Generate localized HTML code based on the input webpage image, the provided HTML DOM tree, and the corresponding bbox information. The priority is to ensure that the generated webpage is as close as possible to the original in terms of structure, style, and content. At the same time, strive to keep the generated HTML DOM tree consistent with the input (preserving the hierarchy and node types as much as possible).
METHOD
Focus on refining the attributes (e.g., styles) of the existing DOM tree or adding new child nodes and attributes where necessary. You may utilize bbox information to divide the webpage into localized sections, applying a divide-and-conquer approach to code generation.
RESPONSE
Provide localized HTML code with inline CSS that ensures the functionality of the module, while preserving the DOM tree structure and enhancing its appearance.
Initialize
In the next message, I will send you a webpage image, module name, the initial HTML DOM tree, and bbox information. Please generate the corresponding HTML code following the rules outlined above.""")

class AgentOptimize(Agent):
    def __init__(self):
        super().__init__("""你是一个擅长于搭建网页的网页工程师。
# CONTEXT #
我想实现一个将网页实现图片转换为实现该网页效果代码的项目。目前交给你的工作是参考网页图片，把已经生成的网页代码进行调整和优化。
# OBJECTIVE
根据输入的网页图片和低质量网页代码，生成高质量的HTML代码。
# RESPONSE #
给出能够和参考图片样式和布局保持高度一直的HTML代码。
# Initialize #
接下来的消息我会给你发送网页图片和已有的网页代码，收到后请按照以上规则给出HTML代码""")
        
class AgentOptimizeStable(Agent):
    def __init__(self):
        super().__init__("""你是一个擅长于搭建网页的网页工程师。
# CONTEXT #
我想实现一个将网页实现图片转换为实现该网页效果代码的项目。目前交给你的工作是参考网页图片，在不改变原网页代码的DOM树结构的基础，把已经生成的网页代码的样式进行调整和优化。
# OBJECTIVE
1. 根据输入的网页图片和原HTML代码，对样式进行调整和优化
2. 严禁修改原网页代码中的DOM树节点类型和层级结构!!!
3. 原封不动地保留原DOM树节点
4. 不改变代码中的图片src
# RESPONSE #
在不改变原HTML代码中DOM树结构的前提下，调整和优化CSS样式
# Initialize #
接下来的消息我会给你发送网页图片和已有的网页代码，收到后请按照以上规则调整和优化原有HTML代码中的CSS""")
        
    def parser(self, text):
        if '```html' in text:
            text = text.split('```html')[1].split('```')[0]
        return text

class AgentAssemble(Agent):
    def __init__(self):
        super().__init__("""# Role #
你是一个擅长于搭建网页的网页工程师。
# CONTEXT #
我想实现一个将网页实现图片转换为实现该网页效果代码的项目。目前交给你的工作是根据网页整体，划分后的模块名称，图片区域和对应的HTML代码，组合生成完整的网页HTML代码。
# OBJECTIVE
根据网页整体，划分后的模块名称，图片区域和对应的HTML代码，组合生成完整的网页HTML代码。
# RESPONSE #
给出能够实现整体功能的HTML网页代码。
# Initialize #
接下来的消息我会给你发送网页整体图片和模块名称，图片区域和对应的HTML代码，收到后请按照以上规则给出HTML代码""")    