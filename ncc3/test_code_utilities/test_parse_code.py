import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from ncc3.utils.code_util.apex.apex_code_util import ApexCodeUtility

apex_code_util = ApexCodeUtility()

sample_code = """
    public class SampleClass {    
        public Integer myNumber;
        public Integer getMyNumber() {
            // Return the current value of myNumber
            return this.myNumber;
        }
    }
"""

ast = apex_code_util.parse(sample_code)

# This will print the tree-sitter AST object
print(ast)