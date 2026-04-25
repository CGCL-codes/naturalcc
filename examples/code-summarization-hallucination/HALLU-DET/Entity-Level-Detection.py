import os
import json
import random
import time
import logging
import re
import ast 
from typing import List, Dict, Any, Optional

MODEL_NAME = "gemini-2.5-flash"
try:
    import javalang
    from javalang.tree import ( 
        MethodDeclaration, ClassDeclaration, InterfaceDeclaration, EnumDeclaration,
        VariableDeclarator, FormalParameter, Type, MethodInvocation, MemberReference
    )
    from javalang.tokenizer import LexerError 
    from javalang.parser import JavaParserError
    JAVALANG_AVAILABLE = True
except ImportError:
    javalang = None
    JAVALANG_AVAILABLE = False
    logging.error("javalang library or specific components not found. Please install it: pip install javalang", exc_info=True) 
except Exception as e: 
    javalang = None
    JAVALANG_AVAILABLE = False
    logging.error(f"An unexpected error occurred during javalang import: {e}", exc_info=True)

try:
    import google.generativeai as genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GOOGLE_GENAI_AVAILABLE = False
    logging.error("google-generativeai library not found. Please install it: pip install google-generativeai")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_KEY = "API KEY HERE"

gemini_model_instance = None
if GOOGLE_GENAI_AVAILABLE:
    if not API_KEY or API_KEY == "xxxx":
        logging.error("GEMINI_API_KEY environment variable not set or is placeholder 'xxxx'.")
    else:
        try:
            genai.configure(api_key=API_KEY)
            gemini_model_instance = genai.GenerativeModel(MODEL_NAME)
            logging.info("Gemini API configured successfully with gemini-2.5-flash.")
        except Exception as e:
            logging.error(f"Failed to configure Gemini API: {e}")
else:
    logging.error("Gemini library not available.")


def call_gemini_api(prompt: str, task_description: str, retries=100, delay=200) -> Optional[str]:
    if not gemini_model_instance:
        logging.error(f"Gemini model not initialized. Cannot perform {task_description}.")
        return None

    for attempt in range(retries):
        try:
            logging.info(f"Calling Gemini for {task_description} (Attempt {attempt+1})...")
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            response = gemini_model_instance.generate_content(prompt, safety_settings=safety_settings)
            if hasattr(response, 'text'):
                return response.text.strip()
            elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 logging.warning(f"Gemini call for {task_description} blocked. Reason: {response.prompt_feedback.block_reason}")
                 return None 
            else:
                 logging.warning(f"Gemini response for {task_description} did not contain text or block reason. Response parts: {response.parts}")
                 return None
        except Exception as e:
            logging.error(f"Error calling Gemini API for {task_description} (Attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    logging.error(f"Max retries reached for {task_description}. Failed to get response.")
    return None

def call_gemini_ner_on_summary(summary: str) -> List[str]:
    prompt = f"""
    Analyze the following Java code summary and extract potential code-related entities.
    Focus on identifying  method names, variable names and parameter names mentioned in the summary.

    Output *only* a Python-parseable list of strings, where each string is a unique identified entity.
    Example output: ['calculateValue', 'inputValue', 'resultsList']

    Summary:
    \"\"\"
    {summary}
    \"\"\"

    Extracted Entities List:
    """
    raw_response = call_gemini_api(prompt, "NER")
    if raw_response is None: return []

    cleaned_str = raw_response
    if cleaned_str.startswith("```python"): cleaned_str = cleaned_str[len("```python"):].strip()
    if cleaned_str.startswith("```"): cleaned_str = cleaned_str[len("```"):].strip()
    if cleaned_str.endswith("```"): cleaned_str = cleaned_str[:-len("```")].strip()

    try:
        entities = ast.literal_eval(cleaned_str)
        if isinstance(entities, list) and all(isinstance(e, str) for e in entities):
            logging.info(f"Gemini NER identified entities: {entities}")
            cleaned_entities = {apply_string_matching_heuristics(e) for e in entities if apply_string_matching_heuristics(e)}
            return sorted(list(cleaned_entities))
        else:
            logging.warning(f"Gemini NER output (after cleaning) was not a list of strings: {cleaned_str}")
            return []
    except (ValueError, SyntaxError, TypeError) as parse_error:
        logging.warning(f"Failed to parse Gemini NER output list: {parse_error}. Cleaned Output was: {cleaned_str}")
        return []

def call_gemini_content_extract(entity: str, holistic_summary: str, summary_claims) -> str:
    ans = ""
    claims = [{"id":idx,"text":x["text"]} for idx,x in enumerate(summary_claims)]
    prompt = f"""
    Assume you are an expert in understanding JAVA code.
    You are given with 2 things:
    1. The holistic summary.
    2. A list, in which each item is a sentence in the original summary. The format of each entry in the list is ("id":id, "text": text), in which the id is the id of the sentence and the text is the sentence itself.

    Given the holistic summary, for each sentence in the list, your task is to determine whether this sentence is related to the entity {entity}.
    
    Example: If you think the sentences with id=0 and id=2 in the list is related to the entity, your output should be [0,2]
    Important! You have to control your output style: You should only output a list!
    There will be No explaination or reasoning content in your answer.

    Here is the holistic summary:
    {holistic_summary}

    Here is the list containing sentences:
    {claims}
    """
    response_text = call_gemini_api(prompt, f"extracting '{entity}'")
    import ast
    lst = ast.literal_eval('[' + response_text.strip().split('[',1)[1].rsplit(']',1)[0] + ']')
    for x in lst:
        try:
            ans = ans + " " + claims[x]["text"]
        except Exception as e:
            continue
    return ans

def call_gemini_intent_verification(entity: str, intent_context: str, code: str) -> str:

    prompt = f"""
    Assume you are an expert in understanding JAVA code.
    Your task is to verify whether the description of '{entity}' in the given text is hallucinated, grounded, or irrelevant with respect to the code.
    Only output one of the following labels: ["GROUNDED", "HALLUCINATED", "IRRELEVANT"].

    Description:
    {intent_context}

    [CODE]
    ```java
    {code}
    ```
    [/CODE]

    Label:
    """
    response_text = call_gemini_api(prompt, f"Intent Verification for '{entity}'")

    if response_text:
        label = response_text.strip().upper().replace('"', '').replace('[', '').replace(']', '')
        if label not in ["HALLUCINATED", "GROUNDED", "IRRELEVANT"]:
            def extract_label(response):
                matches = re.findall(r'(correct|incorrect|irrelevant)', response, flags=re.IGNORECASE)
                if not matches:
                    return None  
                return matches[-1].upper()
            try:
                label = extract_label(response_text)
            except Exception as e:
                print("\n" * 10 + "there is something wrong")
                print(response_text)
                print("----------------" + "\n" * 10)
        if label in ["HALLUCINATED", "GROUNDED", "IRRELEVANT"]:
            logging.info(f"Gemini Intent Verification for '{entity}': {label}")
            return label
        else:
            logging.warning(f"Gemini Intent Verification returned unexpected label: {label}. Defaulting to IRRELEVANT.")
            return "IRRELEVANT"
    else:
         logging.warning(f"Gemini Intent Verification failed for '{entity}'. Defaulting to IRRELEVANT.")
         return "IRRELEVANT" 


def apply_string_matching_heuristics(entity: str) -> str:
    if not entity: return ""
    entity = re.sub(r'(\(\)|\[\])$', '', entity) 
    entity = entity.strip('\'"` ')
    return entity


class CodeAnalyzer:
    def __init__(self):
        if not JAVALANG_AVAILABLE:
            raise ImportError("javalang library is required for CodeAnalyzer.")

    def _extract_entities_from_code_javalang(self, code: str) -> set[str]:
        entities = set()
        is_wrapped = False
        try:
            logging.debug("Attempting to parse Java code with javalang...")
            if not ("class " in code or "interface " in code or " enum " in code):
                 code = f"class DummyWrapper {{\n{code}\n}}"
                 is_wrapped = True
                 logging.debug("Wrapped code snippet in DummyWrapper for parsing.")

            tree = javalang.parse.parse(code)
            logging.debug("Java code parsed successfully.")

            for _, node in tree:
                 node_name = None
                 if isinstance(node, (ClassDeclaration, InterfaceDeclaration, EnumDeclaration, MethodDeclaration, VariableDeclarator, FormalParameter)):
                     if hasattr(node, 'name') and node.name:
                         node_name = node.name
                 elif isinstance(node, Type):
                      if node.name and not isinstance(node.name, list):
                          primitives = {'byte', 'short', 'int', 'long', 'float', 'double', 'boolean', 'char', 'void', 'String', 'Object'}
                          if node.name in primitives or node.name[0].isupper():
                              node_name = node.name
                 if node_name and not (is_wrapped and node_name == "DummyWrapper"):
                     entities.add(node_name)

        except (LexerError, JavaParserError, TypeError, Exception) as e:
            logging.warning(f"Failed to parse Java code with javalang: {e.__class__.__name__}: {e}")
            return set()

        logging.info(f"[CodeAnalyzer] Extracted code entities: {entities}")
        return entities

    def analyze(self, code_snippet: str, language: str) -> Dict[str, Any]:
        if language.lower() != 'java':
            logging.error(f"CodeAnalyzer currently only supports Java via javalang. Language received: {language}")
            return {"code_entities": [], "ast_root": None}

        code_facts = {
            "code_entities": [],
            "ast_root": None 
        }
        entities = self._extract_entities_from_code_javalang(code_snippet)
        code_facts["code_entities"] = sorted(list(entities))


        return code_facts

def my_split(summary: str) -> List[str]:
    lst = re.split(r'(?<=[.!?])\s+|(?<=:)\s*\n\s*', summary)

    sentences = []
    for index,x in enumerate(lst):
        if len(x) > 7 or index == len(lst) - 1:
            sentences.append(x)
            continue
        lst[index + 1] = x + " " + lst[index + 1]
    return sentences
    
class SummaryAnalyzer:
    def analyze(self, summary: str) -> List[Dict[str, Any]]:
        logging.info(f"[SummaryAnalyzer] Analyzing summary: '{summary[:50]}...'")
        claims = []
        summary_entities_list = call_gemini_ner_on_summary(summary)
        summary_entities_set = set(summary_entities_list)

        lst = re.split(r'(?<=[.!?])\s+|(?<=:)\s*\n\s*', summary)

        sentences = []
        for index,x in enumerate(lst):
            if len(x) > 7 or index == len(lst) - 1:
                sentences.append(x)
                continue

            lst[index + 1] = x + " " + lst[index + 1]

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence: continue
            sentence_entities = []
            for entity in summary_entities_set:
                 entity_pattern = r'\b' + re.escape(entity) + r'\b'
                 if re.search(entity_pattern, sentence, re.IGNORECASE):
                     sentence_entities.append(entity)

            claims.append({
                 "id": f"claim_{i}",
                 "text": sentence,
                 "entities_mentioned": sorted(list(set(sentence_entities))) 
            })
        logging.info(f"[SummaryAnalyzer] Extracted claims (sentence level) with entities: {len(claims)}")
        return claims

class DirectGroundingVerifier:

    def _match_entities(self, code_entities: set[str], summary_entities: set[str]) -> tuple[set[str], set[str]]:
        mapped_entities = set()
        unmapped_entities = set()
        normalized_code_map = {apply_string_matching_heuristics(e).lower(): e for e in code_entities}

        for summary_entity in summary_entities:
            norm_summary_entity = apply_string_matching_heuristics(summary_entity).lower()
            if not norm_summary_entity: continue

            if norm_summary_entity in normalized_code_map:
                mapped_entities.add(normalized_code_map[norm_summary_entity]) 
            else:
               
                is_already_mapped = any(apply_string_matching_heuristics(m).lower() == norm_summary_entity for m in mapped_entities)
                if not is_already_mapped:
                    unmapped_entities.add(summary_entity) 

        
        return mapped_entities, unmapped_entities

    def _extract_intent_context(self, entity: str, summary_claims: List[Dict]) -> str:
        relevant_sentences = []
        entity_base = re.escape(apply_string_matching_heuristics(entity))
        entity_pattern = r'\b' + entity_base + r'(?:\(\))?' + r'\b'

        for claim in summary_claims:
            if re.search(entity_pattern, claim["text"], re.IGNORECASE):
                 relevant_sentences.append(claim["text"])
        return " ".join(relevant_sentences)

    def verify(self, summary_claims: List[Dict], code_facts: Dict, code_snippet: str, holistic_summary: str) -> List[Dict]:
        flags = []
        code_entities_set = set(code_facts.get("code_entities", []))

        all_summary_entities = set()
        for claim in summary_claims:
            all_summary_entities.update(claim.get("entities_mentioned", []))
        logging.info(f"[HALLU-DET] Total unique entities mentioned in summary: {len(all_summary_entities)}")

        mapped_entities, unmapped_entities = self._match_entities(code_entities_set, all_summary_entities)

        for claim in summary_claims:
            claim_flags = {"claim_id": claim["id"], "flags": []}
            claim_unmapped = set(claim.get("entities_mentioned", [])) & unmapped_entities


        logging.info(f"[HALLU-DET] Starting Intent Verification for {len(mapped_entities)} mapped entities...")
        incorrect_intent_entities = set()
        is_hallucinated = False
        for entity in mapped_entities:
            summary_of_entity = call_gemini_content_extract(entity, holistic_summary, summary_claims)
            verification_label = call_gemini_intent_verification(entity, summary_of_entity, code_snippet)

            if verification_label == "HALLUCINATED":
                logging.info(f"[HALLU-DET] HALLUCINATED intent detected for entity: {entity}")
                incorrect_intent_entities.add(entity)
                is_hallucinated = True
                break

        if incorrect_intent_entities:
             for claim in summary_claims:
                 claim_flags = next((f for f in flags if f["claim_id"] == claim["id"]), None)
                 if claim_flags is None: 
                      claim_flags = {"claim_id": claim["id"], "flags": []}
                      needs_append = True
                 else:
                      needs_append = False 

                 mentioned_incorrect = set(claim.get("entities_mentioned", [])) & incorrect_intent_entities
                 if mentioned_incorrect:
                     logging.info(f"[HALLU-DET Claim '{claim['id']}' relates to entities with incorrect intent: {mentioned_incorrect}")
                     if not any(f['type'] == 'FACT_CONTRADICTED' for f in claim_flags['flags']):
                          claim_flags["flags"].append({
                              "type": "FACT_CONTRADICTED",
                              "details": f"Context describing entities '{', '.join(sorted(list(mentioned_incorrect)))}' deemed incorrect by LLM."
                          })
                     if needs_append and claim_flags["flags"]:
                          flags.append(claim_flags)
        return flags,is_hallucinated


class HallucinationDetector:
    def __init__(self):
        if not JAVALANG_AVAILABLE:
            raise ImportError("HALLU-DET requires javalang for Java code analysis.")
        if not GOOGLE_GENAI_AVAILABLE or not gemini_model_instance:
             logging.warning("Gemini not available/configured. LLM-based components (NER, Intent, Verifier B) will be disabled.")

        self.code_analyzer = CodeAnalyzer()
        self.summary_analyzer = SummaryAnalyzer()
        self.direct_verifier = DirectGroundingVerifier()


    def detect(self, code_snippet: str, language: str, generated_summary: str) -> Dict[str, Any]:
        if language.lower() != 'java':
             logging.error("This implementation currently only supports Java.")
             return {"decision": "ERROR", "flagged_claims": [], "error": "Unsupported language"}
        language = 'java' 
        print(f"\n--- Starting Hallucination Detection for Java ---")
        start_time = time.time()

        
        code_facts = self.code_analyzer.analyze(code_snippet, language)


        summary_claims = self.summary_analyzer.analyze(generated_summary)
        if not summary_claims:
             logging.warning("No claims extracted from summary.")
             return { "decision": "LIKELY_FACTUAL", "flagged_claims": [], "error": "No claims extracted."}

        all_flags = {} 
        is_hallucinated = False
     

        flags_a,is_hallucinated = self.direct_verifier.verify(summary_claims, code_facts, code_snippet, generated_summary)

        for flag_info in flags_a:

            existing_flags = all_flags.setdefault(flag_info["claim_id"], [])
            for new_flag in flag_info["flags"]:
                if new_flag not in existing_flags: 
                        existing_flags.append(new_flag)

        final_decision = "LIKELY_FACTUAL"
        flagged_claims_output = []
        critical_flags = {"FACT_CONTRADICTED"} 

        num_claims = len(summary_claims)
        claims_with_critical_error = 0

        for claim_id, flags in all_flags.items():
            if flags:
                 claim_text = next((c["text"] for c in summary_claims if c["id"] == claim_id), "Claim text not found")
                 flagged_claims_output.append({"claim_id": claim_id, "claim_text": claim_text, "flags": flags})

                 has_critical = False
                 for f in flags:
                     if f["type"] in critical_flags:
                         has_critical = True

                 if has_critical:
                     claims_with_critical_error += 1
                     final_decision = "HALLUCINATION_DETECTED" 

        reason = "No hallucinations detected."
        if final_decision == "HALLUCINATION_DETECTED":
             crit_flags_found = {f['type'] for item in flagged_claims_output for f in item['flags'] if f['type'] in critical_flags}
             reason = f"Hallucination detected due to: {', '.join(sorted(list(crit_flags_found)))}."

        end_time = time.time()

        if is_hallucinated == True:
            final_decision = "HALLUCINATION_DETECTED"
        return {
            "decision": final_decision,
            "reason": reason,
            "flagged_claims": flagged_claims_output
        }

import itertools

if __name__ == "__main__":
    INPUT_FILE = "INPUTFILE"
    OUTPUT_FILE = "OUTPUTFILE"
  
    BEGIN_LINE = 0
    END_LINE = None

    index = []
    ans = []

    if not JAVALANG_AVAILABLE or not GOOGLE_GENAI_AVAILABLE or not gemini_model_instance:
         print("\nPrerequisites not met (javalang, google-generativeai, or Gemini model init failed). Cannot run examples.")
    else:
        detector = HallucinationDetector()

        total_sample = 0
        hallucination_cnt = 0
        with open(INPUT_FILE,"r") as input_file,open(OUTPUT_FILE,"a") as output_file:
            for idx,line in enumerate(itertools.islice(input_file,BEGIN_LINE,END_LINE)):
                print(f"\nnow we are begin line {idx + BEGIN_LINE}", flush=True)
                total_sample += 1
                p = json.loads(line)
                code_ex = p["code"]
                summary_ex = p["summary"]
                result_ex = detector.detect(code_ex, "java", summary_ex)

                if result_ex["decision"] == "HALLUCINATION_DETECTED":
                    print("OH NO, THERE IS A HALLUCINATION")
                else:
                    print("Good, the summary is grounded")

                decision = (result_ex["decision"] != "HALLUCINATION_DETECTED")
                
                if "ground_truth" in p:
                    vi = {"index":idx + BEGIN_LINE,"code":code_ex,"res":result_ex,"decision":decision,"ground_truth":p["ground_truth"],"summary":summary_ex}
                else:
                    vi = {"index":idx + BEGIN_LINE,"code":code_ex,"res":result_ex,"decision":decision,"summary":summary_ex}

                json.dump(vi,output_file)
                output_file.write('\n')
                