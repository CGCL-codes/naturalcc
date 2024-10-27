import openai
import pandas as pd
import time
import re
import sys

openai.api_key = 'your key'


def evaluate(model, cot, rating_form, reference):

    criteria = {
           "Coherence": "The summary should exhibit clear structural organization, progressing logically from sentence "
                        "to sentence to form a coherent body of information about the topic.",
           "Consistency": "Evaluating the alignment of facts between the summary and the code snippet. A consistent "
                          "summary should contain only statements supported by the source code, while penalizing any "
                          "inclusion of hallucinated facts.",
           "Fluency": "Assessing the quality of each sentence. Sentences should be free from repetition, formatting "
                      "issues, capitalization errors, or clear grammatical problems (e.g., fragments) that affect "
                      "readability.",
           "Relevance": "Evaluating the selection of vital content from the source code. The summary should include only "
                        "essential information from the source document, with penalties for redundancies and excessive "
                        "details.",
           # "Coherence": "the summary should be well-structured and well-organized. The summary should not just be a heap "
        #              "of related information, but should build from sentence to sentence to a coherent body of "
        #              "information about a topic.",
        #
        # "Consistency": "the factual alignment between the summary and the summarized code. A factually consistent "
        #                "summary contains only statements that are entailed by the source code. Annotators were "
        #                "also asked to penalize summaries that contained hallucinated facts. ",
        #
        # "Fluency": "the quality of individual sentences. The sentence should have no repetitive word, formatting "
        #            "problems, capitalization errors or obviously ungrammatical sentences ( "
        #            "e.g., fragments, missing components) that make the text difficult to understand.",
        #
        # "Relevance": "selection of important content from the source. The summary should include only important "
        #              "information from the source document. Annotators were instructed to penalize summaries that "
        #              "contained redundancies and excess information.",
    }

    evaluation_step = {
        'Coh': '1. Read the source code carefully and understand its main functionality and key operations.'
               '2. Read the code comments and compare them to the source code. Check if the comments accurately describe'
               'the main functionality and key operations of the code, and if they present them in a clear and '
               'logical order. '
               '3. Assign a score for coherence on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
               'based on the Evaluation Criteria. ',
        'Con': '1. Read the Source Code carefully and understand its main functionality and any key operations.'
               '2. Read the code comments and compare them to the source code to evaluate its factual alignment.'
               'Ensure that the summary contains only statements that are present or implied in the source code.'
               'Be on the lookout for any hallucinated facts or information in the summary that isn\'t supported by the'
               'source code. If any are found, they should be penalized in your evaluation.'
               '3. Assign a score for consistency on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
               'based on the Evaluation Criteria. ',
        'Flu': '1. Read the code comments carefully and examine each sentence to ensure it is grammatically correct.'
               '2. Identify any glaring grammatical errors, such as sentence fragments, missing components like verbs or subjects, or any other issue that makes the text difficult to understand '
               '3. Check for any instances of repetitive words that can hamper clarity and ensure proper capitalization throughout the comments.'
               '4. Assign a score for fluency on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
               'based on the Evaluation Criteria. ',
        'Ref': '1. Read the source code carefully and understand its key information and primary actions of the code.'
               '2. Read the code comments and compare them to the source code. '
               'Evaluate the completeness of the main information. The summary should provide a complete explanation of the main information without omitting significant details.'
               '3. Check if the code comments include repetitive or unnecessary information. '
               'Annotators should be vigilant about penalizing summaries that deviate from the source code\'s primary intent by including tangential or redundant data.'
               '4. Assign a score for reference on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
               'based on the Evaluation Criteria. ',
    }

    rating = ''
    if rating_form == 0:
        rating = 'Evaluation Form (scores ONLY):'
    elif rating_form == 1:
        rating = 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
    elif rating_form == 2:
        rating = 'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'

    if reference:
        roles = {
            # coherence
            "Original Code Author": "As the Original Code Author, having written the code, you ensure the coherence of the "
                             "code summary, ensuring that it clearly conveys the main logic of the code.",
            # Consistency
            "Code Reviewer1": "As a Code Reviewer, serving as an experienced developer, you guarantee that the summary "
                                    "remains consistent with the original code. You ensure that the summary captures the "
                                    "primary functionality and logic of the code without introducing any additional or "
                                    "unrelated content.",
            # Fluency
            "Code Reviewer2": "As a Code Reviewer, serving as an experienced developer, you focus on ensuring that the summary is written smoothly, with clear "
                           "sentences and appropriate wording. You challenge other judgments and provide alternative "
                           "solutions when necessary.",
            # Relevance
            "Code Editor": "As a Code Editor, concentrating on the business or functional relevance of the code, "
                               "you ensure that the summary captures the key significance of the code in the larger "
                               "system or project.",
        }
        if rating_form == 1:
            example = {
                'Coh': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary is coherent and concise, effectively summarizing the code\'s main action in a clear, structured sentence.'
                       'Rating: 4'
                       'Source Code: public tsactiondelay ( transit section action tsa , int delay ) { tsa = tsa ; delay = delay ; }'
                       'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary:  return a deals which is , a is optionally it not or equal + + equal it not not not not ? a '
                       'specified , not not . equal can equal ; ; a ; a dispatcher is is , . . . . . . . . .'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary is incoherent and unrelated to the source code, which describes a constructor for tsactiondelay.'
                       'Rating: 0'
                       'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       'Reference Summary: methods for starting asynchronous execution .'
                       'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary begins accurately but is marred by irrelevant repetition of words.'
                       'Rating: 2'
                       'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
                       'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                       'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary starts with a relevant description but devolves into a repetitive and irrelevant listing of numbers and the word \'minscale\', losing coherence and accuracy in relation to the source code.'
                       'Rating: 1',
                'Con': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary "creates a new zip entry with the name" is factually consistent with the source code, which describes a method for creating a zip entry with a given name. There are no hallucinated facts or inconsistencies in the summary.'
                       'Rating: 4'
                       'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                       'Reference Summary: remove a scanning callback .'
                       'Summary: if a scanning and .'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary "if a scanning and ." is incomplete and does not accurately convey the primary action of the source code, which is to remove a scanning callback. It lacks clarity and does not align well with the reference summary or the code\'s functionality.'
                       'Rating: 1'
                       'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       'Summary: creates a new dexportprivatekeypvk dialog .'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary "creates a new dexportprivatekeypvk dialog" is inconsistent with the source code, which specifically mentions dexportprivatekeyopenssl. This indicates a factual misalignment between the summary and the code.'
                       'Rating: 0'
                       'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
                       'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                       'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary begins with a relevant phrase, "define a omraster lat lon," which aligns with the source code\'s creation of an omscalingraster object. However, the repetition of "lat," "lon," "minscale," and the sequence of "4000000" does not correspond to any factual content in the source code and introduces inconsistency.'
                       'Rating: 3',
                'Flu': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary "creates a new zip entry with the name" is fluent. It is grammatically correct, free from repetitive words, formatting problems, and capitalization errors, making it easy to understand.'
                       'Rating: 4'
                       'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                       'Reference Summary:  remove a scanning callback .'
                       'Summary: if a scanning and .'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary "if a scanning and ." is not fluent. It is grammatically incorrect, and appears to be an incomplete sentence fragment, making it difficult to understand.'
                       'Rating: 0'
                       'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       'Summary:  creates a new dexportprivatekeypvk dialog .'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary "creates a new dexportprivatekeypvk dialog" is fluent in terms of its structure and grammar. Despite the mismatch with the specific class name in the source code, the sentence itself is well-formed, without repetitive words or grammatical errors.'
                       'Rating: 4'
                       'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
                       'Reference Summary: modifies the given file in place .'
                       'Summary: modifies the the file given . for prefixes . the file'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary "modifies the the file given . for prefixes . the file" suffers from fluency issues. The repetition of "the" and fragmented phrases like "for prefixes . the file" result in a grammatically incorrect and hard-to-understand sentence.'
                       'Rating: 0',
                'Ref': 'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       'Summary:  creates a new dexportprivatekeypvk dialog .'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: the summary "parses the elements and, the store. the." is disjointed and lacks coherence, failing to capture the essential function of the source code. The summary does not convey the specific operation of adjusting the event start date based on the recurrence rule, as detailed in the reference summary.'
                       'Rating: 0'
                       'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
                       'Reference Summary:  invoked if the computation completes successfully'
                       'Summary: invoked completes a computation successfully successfully'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: the summary "invoked completes a computation successfully successfully" captures the essence of the source code, which is about a method being invoked upon successful completion of a computation. However, the redundancy in the use of "successfully" could be seen as a minor issue, as it introduces a slight irrelevance through repetition.'
                       'Rating: 4'
                       'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
                       'Reference Summary: modifies the given file in place .'
                       'Summary: modifies the the file given . for prefixes . the file'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: the summary "modifies the the file given. for prefixes. the file" partially captures the essence of the source code, which is about modifying a file. However, the phrasing "for prefixes. the file" and the repetition of "the" add irrelevant and unclear elements, reducing its relevance.'
                       'Rating: 2'
                       'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       'Reference Summary: methods for starting asynchronous execution .'
                       'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis:  the initial part of the summary, "methods for starting asynchronous execution," aligns well with the source code\'s function of initiating asynchronous processes. However, the repeated phrases "process process process process parent parent" introduce irrelevant and redundant information that does not add value or clarity, thereby reducing the overall relevance.'
                       'Rating: 3',
            }
        elif rating_form == 2:
            example = {
                'Coh': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 4'
                       'Rationale: The summary is coherent, succinctly conveying the main action of the source code without any disorganized or extraneous information.'
                       'Source Code: public tsactiondelay ( transit section action tsa , int delay ) { tsa = tsa ; delay = delay ; }'
                       'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary:  return a deals which is , a is optionally it not or equal + + equal it not not not not ? a '
                       'specified , not not . equal can equal ; ; a ; a dispatcher is is , . . . . . . . . .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 0'
                       'Rationale: The summary is incoherent.'
                       'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       'Reference Summary: methods for starting asynchronous execution .'
                       'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 2'
                       'Rationale: While the initial part of the summary, "methods for starting asynchronous execution," is coherent, the latter part with repeated phrases like "process process process process parent parent" disrupts the coherence, introducing a disjointed and repetitive structure that detracts from the overall clarity and organization of the summary.'
                       'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
                       'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                       'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 1'
                       'Rationale: The summary starts with a coherent phrase "define a omraster lat lat, lon lon lon imageicon,". However, the subsequent repetitive and irrelevant phrases like "scale minscale minscale..." and the series of "4000000" significantly disrupt the coherence.',
                'Con': 'Rationale: The summary "if a scanning and ." lacks factual consistency with the source code. While it hints at a conditional operation involving scanning, it fails to accurately or fully convey the main action of the source code, which is the removal of a scanning callback.'
                       'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       'Summary: creates a new dexportprivatekeypvk dialog .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 0'
                       'Rationale: The summary "creates a new dexportprivatekeypvk dialog" is factually inconsistent with the source code, which is about creating a dexportprivatekeyopenssl dialog. The summary incorrectly references a different dialog (dexportprivatekeypvk), which is not mentioned or implied in the source code.'
                       'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }\n'
                       'Reference Summary:  invoked if the computation completes successfully'
                       'Summary: invoked completes a computation successfully successfully'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 4'
                       'Rationale: The summary "invoked completes a computation successfully successfully" is factually consistent with the source code. The source code describes a method that is invoked upon the successful completion of a computation. '
                       'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }\n'
                       'Reference Summary: modifies the given file in place .'
                       'Summary: modifies the the file given . for prefixes . the file'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 2'
                       'Rationale: The summary "methods for starting asynchronous execution. process process process process parent parent" begins correctly and is consistent with the source code\'s description of starting asynchronous execution. However, the latter part with the repetition of "process" and the addition of "parent parent" introduces unrelated elements that are not entailed by the source code. This decreases the overall factual consistency, as these additional phrases do not align with the specific actions or content of the code.'
                       'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }\n'
                       'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                       'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 3'
                       'Rationale: The summary "define a omraster lat lat, lon lon lon imageicon, scale minscale..." significantly deviates from the source code\'s content. While it starts with a relevant phrase, the addition of repetitive and irrelevant details about "minscale" and a series of "4000000" are not entailed by or related to the source code. This introduces factual inconsistencies and unnecessary information, leading to a low alignment with the source code\'s actual functionality.',
                'Flu': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 4'
                       'Rationale: The summary "creates a new zip entry with the name" is a fluent sentence. It has no repetitive words, formatting problems, capitalization errors, or grammatical issues. The sentence is complete, structurally sound, and easy to understand, meeting the fluency criteria effectively.'
                       'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                       'Reference Summary:  remove a scanning callback .'
                       'Summary: if a scanning and .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 0'
                       'Rationale: It contains incomplete and fragmented sentences, making it difficult to understand. The sentence structure is not clear, and there are issues with grammar.'
                       'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       'Summary:  creates a new dexportprivatekeypvk dialog .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 4'
                       'Rationale: The summary is fluent. It is grammatically correct, clear, and concise, with no repetitive words, formatting issues, or ungrammatical sentences, making it easy to understand. However, there is a mismatch between the summary and the specific class name in the source code, which affects its accuracy.'
                       'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
                       'Reference Summary:  invoked if the computation completes successfully'
                       'Summary: invoked completes a computation successfully successfully'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 0'
                       'Rationale: The summary lacks fluency. It contains repetitive words ("modifies," "the," "given," "file," "the," "file") and lacks clarity. The summary does not effectively convey the meaning of the source code. Additionally, there are grammatical issues in the summary, making it difficult to understand.',
                'Ref': 'Rationale: The summary does not effectively convey the important information from the source code. It lacks relevance and coherence, making it difficult to understand the purpose and functionality of the code. The summary is not in line with the source code\'s content, resulting in a low rating for relevance.'
                       'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       'Summary:  creates a new dexportprivatekeypvk dialog .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 0'
                       'Rationale:  The summary is not relevant to the source code. It provides incorrect information by mentioning a "dexportprivatekeypvk dialog" instead of the expected "dexportprivatekeyopenssl dialog." This inconsistency and inaccuracy result in a low rating for relevance.'
                       'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
                       'Reference Summary:  invoked if the computation completes successfully'
                       'Summary: invoked completes a computation successfully successfully'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 4'
                       'Rationale:The summary is highly relevant to the source code. It accurately conveys that the onSuccess method is invoked when a computation completes successfully. While there is some repetition in the word "successfully," it does not significantly impact the overall relevance of the summary.'
                       'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
                       'Reference Summary: modifies the given file in place .'
                       'Summary: modifies the the file given . for prefixes . the file'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 2'
                       'Rationale: The summary is somewhat relevant to the source code as it mentions the action of modifying a file, but it lacks clarity and conciseness. The repeated words "modifies," "the," and "file" make the summary less fluent and slightly less relevant. Additionally, the use of "for prefixes" is not clear and seems out of place.'
                       'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       'Reference Summary: methods for starting asynchronous execution .'
                       'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 3'
                       'Rationale: The summary is moderately relevant to the source code as it mentions the purpose of the code, which is to start asynchronous execution. However, the repetition of words like "process" and "parent" in the summary reduces its fluency and clarity.',
            }
        else:
            example = {
                'Coh': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (scores ONLY): '
                       '- Coherence: 4'
                       'Source Code: public tsactiondelay ( transit section action tsa , int delay ) { tsa = tsa ; delay = delay ; }'
                       'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary:  return a deals which is , a is optionally it not or equal + + equal it not not not not ? a '
                       'specified , not not . equal can equal ; ; a ; a dispatcher is is , . . . . . . . . .'
                       'Evaluation Form (scores ONLY): '
                       '- Coherence: 0'
                       'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       'Reference Summary: methods for starting asynchronous execution .'
                       'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       'Evaluation Form (scores ONLY): '
                       '- Coherence: 2'
                       'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
                       'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                       'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                       'Evaluation Form (scores ONLY): '
                       '- Coherence: 1',
                'Con': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (scores ONLY): '
                       '- Consistency: 4'
                       'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                       'Reference Summary: remove a scanning callback .'
                       'Summary: if a scanning and .'
                       'Evaluation Form (scores ONLY): '
                       '- Consistency: 1'
                       'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       'Summary: creates a new dexportprivatekeypvk dialog .'
                       'Evaluation Form (scores ONLY): '
                       '- Consistency: 0'
                       'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
                       'Reference Summary:  invoked if the computation completes successfully'
                       'Summary: invoked completes a computation successfully successfully'
                       'Evaluation Form (scores ONLY): '
                       '- Consistency: 4'
                       'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
                       'Reference Summary: modifies the given file in place .'
                       'Summary: modifies the the file given . for prefixes . the file'
                       'Evaluation Form (scores ONLY): '
                       '- Consistency: 2'
                       'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       'Reference Summary: methods for starting asynchronous execution .'
                       'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       'Evaluation Form (scores ONLY): '
                       '- Consistency: 3'
                       'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
                       'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                       'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                       'Evaluation Form (scores ONLY): '
                       '- Consistency: 3',
                'Flu': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (scores ONLY): '
                       '- Fluency: 4'
                       'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                       'Reference Summary:  remove a scanning callback .'
                       'Summary: if a scanning and .'
                       'Evaluation Form (scores ONLY): '
                       '- Fluency: 0'
                       'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       'Summary:  creates a new dexportprivatekeypvk dialog .'
                       'Evaluation Form (scores ONLY): '
                       '- Fluency: 4'
                       'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
                       'Reference Summary:  invoked if the computation completes successfully'
                       'Summary: invoked completes a computation successfully successfully'
                       'Evaluation Form (scores ONLY): '
                       '- Fluency: 0'
                       'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
                       'Reference Summary: modifies the given file in place .'
                       'Summary: modifies the the file given . for prefixes . the file'
                       'Evaluation Form (scores ONLY): '
                       '- Fluency: 0'
                       'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       'Reference Summary: methods for starting asynchronous execution .'
                       'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       'Evaluation Form (scores ONLY): '
                       '- Fluency: 0'
                       'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
                       'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                       'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                       'Evaluation Form (scores ONLY): '
                       '- Fluency: 0',
                'Ref': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (scores ONLY): '
                       '- Relevance: 4'
                       'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                       'Reference Summary:  remove a scanning callback .'
                       'Summary: if a scanning and .'
                       'Evaluation Form (scores ONLY): '
                       '- Relevance: 1'
                       'Source Code: private void check server response code ( httpurlconnection url connection ) throws request failure exception { try { if ( url connection get response code ( ) != <num> ) { throw new request failure exception ( <string> + url connection get response code ( ) + <string> ) ; } } catch ( ioexception e ) { throw new request failure exception ( <string> , e ) ; } }'
                       'Reference Summary: confirms that the omaha server sent back an " ok " code .'
                       'Summary: code that code in the code response .'
                       'Evaluation Form (scores ONLY): '
                       '- Relevance: 0'
                       'Source Code: private void offset start time if necessary ( time start time , time end time , string rrule , calendar event model model ) { if ( rrule == null || rrule is empty ( ) ) { return ; } m event recurrence parse ( rrule ) ; if ( m event recurrence freq != event recurrence weekly ) { return ; } if ( m event recurrence byday length > m event recurrence byday count ) { return ; } int closest weekday = integer max value ; int weekstart = event recurrence day2time day ( m event recurrence wkst ) ; int start day = start time week day ; for ( int i = <num> ; i < m event recurrence byday count ; i ++ ) { int day = event recurrence day2time day ( m event recurrence byday [ i ] ) ; if ( day == start day ) { return ; } if ( day < weekstart ) { day += <num> ; } if ( day > start day && ( day < closest weekday || closest weekday < start day ) ) { closest weekday = day ; } if ( closest weekday == integer max value || closest weekday < start day ) { if ( day < closest weekday ) { closest weekday = day ; } } } if ( closest weekday < start day ) { closest weekday += <num> ; } int days offset = closest weekday - start day ; start time month day += days offset ; end time month day += days offset ; long new start time = start time normalize ( true ) ; long new end time = end time normalize ( true ) ; model m start = new start time ; model m end = new end time ; }'
                       'Reference Summary:  if the recurrence rule is such that the event start date doesn \' t actually fall in one of the recurrences , then push the start date up to the first actual instance of the event .'
                       'Summary:  parses the elements and , the store . the .'
                       'Evaluation Form (scores ONLY): '
                       '- Relevance: 0'
                       'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       'Summary:  creates a new dexportprivatekeypvk dialog .'
                       'Evaluation Form (scores ONLY): '
                       '- Relevance: 0'
                       'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
                       'Reference Summary:  invoked if the computation completes successfully'
                       'Summary: invoked completes a computation successfully successfully'
                       'Evaluation Form (scores ONLY): '
                       '- Relevance: 4'
                       'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
                       'Reference Summary: modifies the given file in place .'
                       'Summary: modifies the the file given . for prefixes . the file'
                       'Evaluation Form (scores ONLY): '
                       '- Relevance: 2'
                       'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       'Reference Summary: methods for starting asynchronous execution .'
                       'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       'Evaluation Form (scores ONLY): '
                       '- Relevance: 3',
            }
    else:
        roles = {
            # coherence
            "Systems Analyst1": "As a Systems Analyst, you ensure the coherence of the "
                             "code summary, ensuring that it clearly conveys the main logic of the code.",
            # Consistency
            "Code Reviewer1": "As a Code Reviewer, serving as an experienced developer, you guarantee that the summary "
                                    "remains consistent with the original code. You ensure that the summary captures the "
                                    "primary functionality and logic of the code without introducing any additional or "
                                    "unrelated content.",
            # Fluency
            "Systems Analyst2": "As a Systems Analyst, you focus on ensuring that the summary is written smoothly, with clear "
                           "sentences and appropriate wording. You challenge other judgments and provide alternative "
                           "solutions when necessary.",
            # Relevance
            "Code Reviewer2": "As a Code Reviewer, serving as an experienced developer, concentrating on the business or functional relevance of the code, "
                                    "you ensure that the summary captures the key significance of the code in the larger "
                                    "system or project.",
        }
        if rating_form == 1:
            example = {
                'Coh': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       # 'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary is coherent and concise, effectively summarizing the code\'s main action in a clear, structured sentence.'
                       'Rating: 4'
                       'Source Code: public tsactiondelay ( transit section action tsa , int delay ) { tsa = tsa ; delay = delay ; }'
                       # 'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary:  return a deals which is , a is optionally it not or equal + + equal it not not not not ? a '
                       'specified , not not . equal can equal ; ; a ; a dispatcher is is , . . . . . . . . .'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary is incoherent and unrelated to the source code, which describes a constructor for tsactiondelay.'
                       'Rating: 0'
                       'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       # 'Reference Summary: methods for starting asynchronous execution .'
                       'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary begins accurately but is marred by irrelevant repetition of words.'
                       'Rating: 2'
                       'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
                       # 'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                       'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary starts with a relevant description but devolves into a repetitive and irrelevant listing of numbers and the word \'minscale\', losing coherence and accuracy in relation to the source code.'
                       'Rating: 1',
                'Con': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       # 'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary "creates a new zip entry with the name" is factually consistent with the source code, which describes a method for creating a zip entry with a given name. There are no hallucinated facts or inconsistencies in the summary.'
                       'Rating: 4'
                       # 'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                       # # 'Reference Summary: remove a scanning callback .'
                       # 'Summary: if a scanning and .'
                       # 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       # 'Analysis: The summary "if a scanning and ." is incomplete and does not accurately convey the primary action of the source code, which is to remove a scanning callback. It lacks clarity and does not align well with the reference summary or the code\'s functionality.'
                       # 'Rating: 1'
                       # 'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       # # 'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       # 'Summary: creates a new dexportprivatekeypvk dialog .'
                       # 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       # 'Analysis: The summary "creates a new dexportprivatekeypvk dialog" is inconsistent with the source code, which specifically mentions dexportprivatekeyopenssl. This indicates a factual misalignment between the summary and the code.'
                       # 'Rating: 0'
                       'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
                       # 'Reference Summary:  invoked if the computation completes successfully'
                       'Summary: invoked completes a computation successfully successfully'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary is mostly accurate but redundant, reflecting the source code\'s successful computation completion.'
                       'Rating: 4'
                       'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
                       # 'Reference Summary: modifies the given file in place .'
                       'Summary: modifies the the file given . for prefixes . the file'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary "modifies the the file given . for prefixes . the file" starts correctly by indicating the modification of a file but becomes unclear and irrelevant with the addition of "for prefixes."'
                       'Rating: 2'
                       # 'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       # # 'Reference Summary: methods for starting asynchronous execution .'
                       # 'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       # 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       # 'Analysis: The initial part of the summary, "methods for starting asynchronous execution," accurately reflects the source code\'s purpose. However, the repetition of "process process process process parent parent" is irrelevant and reduces the overall clarity.'
                       # 'Rating: 3'
                       'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
                       # 'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                       'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary begins with a relevant phrase, "define a omraster lat lon," which aligns with the source code\'s creation of an omscalingraster object. However, the repetition of "lat," "lon," "minscale," and the sequence of "4000000" does not correspond to any factual content in the source code and introduces inconsistency.'
                       'Rating: 3',
                'Flu': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       # 'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary "creates a new zip entry with the name" is fluent. It is grammatically correct, free from repetitive words, formatting problems, and capitalization errors, making it easy to understand.'
                       'Rating: 4'
                       'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                       # 'Reference Summary:  remove a scanning callback .'
                       'Summary: if a scanning and .'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary "if a scanning and ." is not fluent. It is grammatically incorrect, and appears to be an incomplete sentence fragment, making it difficult to understand.'
                       'Rating: 0'
                       'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       # 'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       'Summary:  creates a new dexportprivatekeypvk dialog .'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary "creates a new dexportprivatekeypvk dialog" is fluent in terms of its structure and grammar. Despite the mismatch with the specific class name in the source code, the sentence itself is well-formed, without repetitive words or grammatical errors.'
                       'Rating: 4'
                       # 'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
                       # # 'Reference Summary:  invoked if the computation completes successfully'
                       # 'Summary: invoked completes a computation successfully successfully'
                       # 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       # 'Analysis: The summary "invoked completes a computation successfully successfully" has a grammatical error due to the repetition of the word "successfully." This redundancy makes the sentence awkward and less fluent, impacting its overall readability.'
                       # 'Rating: 0'
                       'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
                       # 'Reference Summary: modifies the given file in place .'
                       'Summary: modifies the the file given . for prefixes . the file'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: The summary "modifies the the file given . for prefixes . the file" suffers from fluency issues. The repetition of "the" and fragmented phrases like "for prefixes . the file" result in a grammatically incorrect and hard-to-understand sentence.'
                       'Rating: 0',
                'Ref': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       # 'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: the summary "creates a new zip entry with the name" effectively captures the most important action of the source code without including any redundancies or excess information. It concisely states the primary functionality of the code, aligning closely with the key content of the source.'
                       'Rating: 4'
                       'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                       # 'Reference Summary:  remove a scanning callback .'
                       'Summary: if a scanning and .'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: the summary "if a scanning and ." fails to capture the essential action of the source code, which is the removal of a scanning callback. The summary is incomplete and does not convey the key information present in the code, leading to a lack of relevance.'
                       'Rating: 1'
                       # 'Source Code: private void check server response code ( httpurlconnection url connection ) throws request failure exception { try { if ( url connection get response code ( ) != <num> ) { throw new request failure exception ( <string> + url connection get response code ( ) + <string> ) ; } } catch ( ioexception e ) { throw new request failure exception ( <string> , e ) ; } }'
                       # # 'Reference Summary: confirms that the omaha server sent back an " ok " code .'
                       # 'Summary: code that code in the code response .'
                       # 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       # 'Analysis: the summary "code that code in the code response." is vague and fails to capture the essential action of the source code, which is checking and validating the server response code. The summary does not effectively convey the specific function of confirming an "ok" response from the server, as indicated in the reference summary.'
                       # 'Rating: 0'
                       # 'Source Code: private void offset start time if necessary ( time start time , time end time , string rrule , calendar event model model ) { if ( rrule == null || rrule is empty ( ) ) { return ; } m event recurrence parse ( rrule ) ; if ( m event recurrence freq != event recurrence weekly ) { return ; } if ( m event recurrence byday length > m event recurrence byday count ) { return ; } int closest weekday = integer max value ; int weekstart = event recurrence day2time day ( m event recurrence wkst ) ; int start day = start time week day ; for ( int i = <num> ; i < m event recurrence byday count ; i ++ ) { int day = event recurrence day2time day ( m event recurrence byday [ i ] ) ; if ( day == start day ) { return ; } if ( day < weekstart ) { day += <num> ; } if ( day > start day && ( day < closest weekday || closest weekday < start day ) ) { closest weekday = day ; } if ( closest weekday == integer max value || closest weekday < start day ) { if ( day < closest weekday ) { closest weekday = day ; } } } if ( closest weekday < start day ) { closest weekday += <num> ; } int days offset = closest weekday - start day ; start time month day += days offset ; end time month day += days offset ; long new start time = start time normalize ( true ) ; long new end time = end time normalize ( true ) ; model m start = new start time ; model m end = new end time ; }'
                       # # 'Reference Summary:  if the recurrence rule is such that the event start date doesn \' t actually fall in one of the recurrences , then push the start date up to the first actual instance of the event .'
                       # 'Summary:  parses the elements and , the store . the .'
                       # 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       # 'Analysis: '
                       # 'Rating: 0'
                       # 'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       # # 'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       # 'Summary:  creates a new dexportprivatekeypvk dialog .'
                       # 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       # 'Analysis: the summary "parses the elements and, the store. the." is disjointed and lacks coherence, failing to capture the essential function of the source code. The summary does not convey the specific operation of adjusting the event start date based on the recurrence rule, as detailed in the reference summary.'
                       # 'Rating: 0'
                       # 'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
                       # # 'Reference Summary:  invoked if the computation completes successfully'
                       # 'Summary: invoked completes a computation successfully successfully'
                       # 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       # 'Analysis: the summary "invoked completes a computation successfully successfully" captures the essence of the source code, which is about a method being invoked upon successful completion of a computation. However, the redundancy in the use of "successfully" could be seen as a minor issue, as it introduces a slight irrelevance through repetition.'
                       # 'Rating: 4'
                       'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
                       # 'Reference Summary: modifies the given file in place .'
                       'Summary: modifies the the file given . for prefixes . the file'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis: the summary "modifies the the file given. for prefixes. the file" partially captures the essence of the source code, which is about modifying a file. However, the phrasing "for prefixes. the file" and the repetition of "the" add irrelevant and unclear elements, reducing its relevance.'
                       'Rating: 2'
                       'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       # 'Reference Summary: methods for starting asynchronous execution .'
                       'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)'
                       'Analysis:  the initial part of the summary, "methods for starting asynchronous execution," aligns well with the source code\'s function of initiating asynchronous processes. However, the repeated phrases "process process process process parent parent" introduce irrelevant and redundant information that does not add value or clarity, thereby reducing the overall relevance.'
                       'Rating: 3',
            }
        elif rating_form == 2:
            example = {
                'Coh': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       # 'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 4'
                       'Rationale: The summary is coherent, succinctly conveying the main action of the source code without any disorganized or extraneous information.'
                       'Source Code: public tsactiondelay ( transit section action tsa , int delay ) { tsa = tsa ; delay = delay ; }'
                       # 'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary:  return a deals which is , a is optionally it not or equal + + equal it not not not not ? a '
                       'specified , not not . equal can equal ; ; a ; a dispatcher is is , . . . . . . . . .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 0'
                       'Rationale: The summary is incoherent.'
                       'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       # 'Reference Summary: methods for starting asynchronous execution .'
                       'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 2'
                       'Rationale: While the initial part of the summary, "methods for starting asynchronous execution," is coherent, the latter part with repeated phrases like "process process process process parent parent" disrupts the coherence, introducing a disjointed and repetitive structure that detracts from the overall clarity and organization of the summary.'
                       'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
                       # 'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                       'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 1'
                       'Rationale: The summary starts with a coherent phrase "define a omraster lat lat, lon lon lon imageicon,". However, the subsequent repetitive and irrelevant phrases like "scale minscale minscale..." and the series of "4000000" significantly disrupt the coherence.',
                'Con': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       # 'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 4'
                       'Rationale: The summary "creates a new zip entry with the name" is factually consistent with the source code, which details the creation of a new zip entry given a string name. '
                       'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                       # 'Reference Summary: remove a scanning callback .'
                       'Summary: if a scanning and .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 1'
                       'Rationale: The summary "if a scanning and ." lacks factual consistency with the source code. While it hints at a conditional operation involving scanning, it fails to accurately or fully convey the main action of the source code, which is the removal of a scanning callback.'
                       'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       # 'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       'Summary: creates a new dexportprivatekeypvk dialog .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 0'
                       'Rationale: The summary "creates a new dexportprivatekeypvk dialog" is factually inconsistent with the source code, which is about creating a dexportprivatekeyopenssl dialog. The summary incorrectly references a different dialog (dexportprivatekeypvk), which is not mentioned or implied in the source code.'
                       'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
                       # 'Reference Summary:  invoked if the computation completes successfully'
                       'Summary: invoked completes a computation successfully successfully'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 4'
                       'Rationale: The summary "invoked completes a computation successfully successfully" is factually consistent with the source code. The source code describes a method that is invoked upon the successful completion of a computation. '
                       'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
                       # 'Reference Summary: modifies the given file in place .'
                       'Summary: modifies the the file given . for prefixes . the file'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 2'
                       'Rationale: The summary "modifies the the file given. for prefixes. the file" partially aligns with the source code\'s functionality of modifying a file. However, the phrase "for prefixes. the file" and the repetition of "the" introduce elements that are not present or implied in the source code, leading to a reduction in factual consistency.'
                       'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       # 'Reference Summary: methods for starting asynchronous execution .'
                       'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 3'
                       'Rationale: The summary "methods for starting asynchronous execution. process process process process parent parent" begins correctly and is consistent with the source code\'s description of starting asynchronous execution. However, the latter part with the repetition of "process" and the addition of "parent parent" introduces unrelated elements that are not entailed by the source code. This decreases the overall factual consistency, as these additional phrases do not align with the specific actions or content of the code.'
                       'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
                       # 'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                       'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 3'
                       'Rationale: The summary "define a omraster lat lat, lon lon lon imageicon, scale minscale..." significantly deviates from the source code\'s content. While it starts with a relevant phrase, the addition of repetitive and irrelevant details about "minscale" and a series of "4000000" are not entailed by or related to the source code. This introduces factual inconsistencies and unnecessary information, leading to a low alignment with the source code\'s actual functionality.',
                'Flu': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       # 'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 4'
                       'Rationale: The summary "creates a new zip entry with the name" is a fluent sentence. It has no repetitive words, formatting problems, capitalization errors, or grammatical issues. The sentence is complete, structurally sound, and easy to understand, meeting the fluency criteria effectively.'
                       'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                       # 'Reference Summary:  remove a scanning callback .'
                       'Summary: if a scanning and .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 0'
                       'Rationale: It contains incomplete and fragmented sentences, making it difficult to understand. The sentence structure is not clear, and there are issues with grammar.'
                       'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       # 'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       'Summary:  creates a new dexportprivatekeypvk dialog .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 4'
                       'Rationale: The summary is fluent. It is grammatically correct, clear, and concise, with no repetitive words, formatting issues, or ungrammatical sentences, making it easy to understand. However, there is a mismatch between the summary and the specific class name in the source code, which affects its accuracy.'
                       'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
                       # 'Reference Summary:  invoked if the computation completes successfully'
                       'Summary: invoked completes a computation successfully successfully'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 0'
                       'Rationale: The summary lacks fluency. It contains repetitive words ("completes" and "successfully") and lacks clarity. The summary does not effectively convey the meaning of the source code. Additionally, there is a grammatical issue in the summary, making it difficult to understand.'
                       'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
                       # 'Reference Summary: modifies the given file in place .'
                       'Summary: modifies the the file given . for prefixes . the file'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 0'
                       'Rationale: The summary lacks fluency. It contains repetitive words ("modifies," "the," "given," "file," "the," "file") and lacks clarity. The summary does not effectively convey the meaning of the source code. Additionally, there are grammatical issues in the summary, making it difficult to understand.',
                'Ref': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                       'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                       'name ; }'
                       # 'Reference Summary: creates a new zip entry with the specified name.'
                       'Summary: creates a new zip entry with the name'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 4'
                       'Rationale: The summary is highly relevant to the source code. It effectively conveys the essential information from the source code, which is the creation of a new zip entry with a specified name. There are no redundancies or excess information in the summary, and it captures the key details accurately.'
                       'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
                       # 'Reference Summary:  remove a scanning callback .'
                       'Summary: if a scanning and .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 1'
                       'Rationale: While the summary mentions the removal of a scanning callback, it lacks clarity and important details about the functionality of the code. It does not effectively convey the important information from the source code, and there is room for improvement in terms of relevance.'
                       'Source Code: private void check server response code ( httpurlconnection url connection ) throws request failure exception { try { if ( url connection get response code ( ) != <num> ) { throw new request failure exception ( <string> + url connection get response code ( ) + <string> ) ; } } catch ( ioexception e ) { throw new request failure exception ( <string> , e ) ; } }'
                       # 'Reference Summary: confirms that the omaha server sent back an " ok " code .'
                       'Summary: code that code in the code response .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 0'
                       'Rationale: The summary lacks relevance as it does not effectively convey the important information from the source code, which is about checking the server response code for an "ok" code. It contains unclear and irrelevant language, resulting in a low rating for relevance.'
                       'Source Code: private void offset start time if necessary ( time start time , time end time , string rrule , calendar event model model ) { if ( rrule == null || rrule is empty ( ) ) { return ; } m event recurrence parse ( rrule ) ; if ( m event recurrence freq != event recurrence weekly ) { return ; } if ( m event recurrence byday length > m event recurrence byday count ) { return ; } int closest weekday = integer max value ; int weekstart = event recurrence day2time day ( m event recurrence wkst ) ; int start day = start time week day ; for ( int i = <num> ; i < m event recurrence byday count ; i ++ ) { int day = event recurrence day2time day ( m event recurrence byday [ i ] ) ; if ( day == start day ) { return ; } if ( day < weekstart ) { day += <num> ; } if ( day > start day && ( day < closest weekday || closest weekday < start day ) ) { closest weekday = day ; } if ( closest weekday == integer max value || closest weekday < start day ) { if ( day < closest weekday ) { closest weekday = day ; } } } if ( closest weekday < start day ) { closest weekday += <num> ; } int days offset = closest weekday - start day ; start time month day += days offset ; end time month day += days offset ; long new start time = start time normalize ( true ) ; long new end time = end time normalize ( true ) ; model m start = new start time ; model m end = new end time ; }'
                       # 'Reference Summary:  if the recurrence rule is such that the event start date doesn \' t actually fall in one of the recurrences , then push the start date up to the first actual instance of the event .'
                       'Summary:  parses the elements and , the store . the .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 0'
                       'Rationale: The summary does not effectively convey the important information from the source code. It lacks relevance and coherence, making it difficult to understand the purpose and functionality of the code. The summary is not in line with the source code\'s content, resulting in a low rating for relevance.'
                       'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
                       # 'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                       'Summary:  creates a new dexportprivatekeypvk dialog .'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 0'
                       'Rationale:  The summary is not relevant to the source code. It provides incorrect information by mentioning a "dexportprivatekeypvk dialog" instead of the expected "dexportprivatekeyopenssl dialog." This inconsistency and inaccuracy result in a low rating for relevance.'
                       'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
                       # 'Reference Summary:  invoked if the computation completes successfully'
                       'Summary: invoked completes a computation successfully successfully'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 4'
                       'Rationale:The summary is highly relevant to the source code. It accurately conveys that the onSuccess method is invoked when a computation completes successfully. While there is some repetition in the word "successfully," it does not significantly impact the overall relevance of the summary.'
                       'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
                       # 'Reference Summary: modifies the given file in place .'
                       'Summary: modifies the the file given . for prefixes . the file'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 2'
                       'Rationale: The summary is somewhat relevant to the source code as it mentions the action of modifying a file, but it lacks clarity and conciseness. The repeated words "modifies," "the," and "file" make the summary less fluent and slightly less relevant. Additionally, the use of "for prefixes" is not clear and seems out of place.'
                       'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
                       # 'Reference Summary: methods for starting asynchronous execution .'
                       'Summary: methods for starting asynchronous execution . process process process process parent parent'
                       'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')'
                       'Rating: 3'
                       'Rationale: The summary is moderately relevant to the source code as it mentions the purpose of the code, which is to start asynchronous execution. However, the repetition of words like "process" and "parent" in the summary reduces its fluency and clarity.',
            }
        else:
              example = {
            'Coh': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                   'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                   'name ; }'
            # 'Reference Summary: creates a new zip entry with the specified name.\n'
                   'Summary: creates a new zip entry with the name'
                   'Evaluation Form (scores ONLY): '
                   '- Coherence: 4'
                   'Source Code: public tsactiondelay ( transit section action tsa , int delay ) { tsa = tsa ; delay = delay ; }'
            # 'Reference Summary: creates a new zip entry with the specified name.\n'
                   'Summary:  return a deals which is , a is optionally it not or equal + + equal it not not not not ? a '
                   'specified , not not . equal can equal ; ; a ; a dispatcher is is , . . . . . . . . .'
                   'Evaluation Form (scores ONLY): '
                   '- Coherence: 0'
                   'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
            # 'Reference Summary: methods for starting asynchronous execution .'
                   'Summary: methods for starting asynchronous execution . process process process process parent parent'
                   'Evaluation Form (scores ONLY): '
                   '- Coherence: 2'
                   'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
            # 'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                   'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                   'Evaluation Form (scores ONLY): '
                   '- Coherence: 1',
            'Con': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                   'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                   'name ; }'
            # 'Reference Summary: creates a new zip entry with the specified name.\n'
                   'Summary: creates a new zip entry with the name'
                   'Evaluation Form (scores ONLY): '
                   '- Consistency: 4'
                   'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
            # 'Reference Summary: remove a scanning callback .'
                   'Summary: if a scanning and .'
                   'Evaluation Form (scores ONLY): '
                   '- Consistency: 1'
                   'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
            # 'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                   'Summary: creates a new dexportprivatekeypvk dialog .'
                   'Evaluation Form (scores ONLY): '
                   '- Consistency: 0'
                   'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
            # 'Reference Summary:  invoked if the computation completes successfully'
                   'Summary: invoked completes a computation successfully successfully'
                   'Evaluation Form (scores ONLY): '
                   '- Consistency: 4'
                   'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
            # 'Reference Summary: modifies the given file in place .'
                   'Summary: modifies the the file given . for prefixes . the file'
                   'Evaluation Form (scores ONLY): '
                   '- Consistency: 2'
                   'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
            # 'Reference Summary: methods for starting asynchronous execution .'
                   'Summary: methods for starting asynchronous execution . process process process process parent parent'
                   'Evaluation Form (scores ONLY): '
                   '- Consistency: 3'
                   'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
            # 'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                   'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                   'Evaluation Form (scores ONLY): '
                   '- Consistency: 3',
            'Flu': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                   'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                   'name ; }'
            # 'Reference Summary: creates a new zip entry with the specified name.\n'
                   'Summary: creates a new zip entry with the name'
                   'Evaluation Form (scores ONLY): '
                   '- Fluency: 4'
                   'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
            # 'Reference Summary:  remove a scanning callback .'
                   'Summary: if a scanning and .'
                   'Evaluation Form (scores ONLY): '
                   '- Fluency: 0'
                   'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
            # 'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                   'Summary:  creates a new dexportprivatekeypvk dialog .'
                   'Evaluation Form (scores ONLY): '
                   '- Fluency: 4'
                   'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
            # 'Reference Summary:  invoked if the computation completes successfully'
                   'Summary: invoked completes a computation successfully successfully'
                   'Evaluation Form (scores ONLY): '
                   '- Fluency: 0'
                   'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
            # 'Reference Summary: modifies the given file in place .'
                   'Summary: modifies the the file given . for prefixes . the file'
                   'Evaluation Form (scores ONLY): '
                   '- Fluency: 0'
                   'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
            # 'Reference Summary: methods for starting asynchronous execution .'
                   'Summary: methods for starting asynchronous execution . process process process process parent parent'
                   'Evaluation Form (scores ONLY): '
                   '- Fluency: 0'
                   'Source Code: public omscalingraster ( double ullat , double ullon , double lrlat , double lrlon , image icon ii ) { this ( ullat , ullon , lrlat , lrlon , ii get image ( ) ) ; }'
            # 'Reference Summary: create an omraster , lat / lon placement with an imageicon .'
                   'Summary: define a omraster lat lat , lon lon lon imageicon , scale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale minscale , 4000000 4000000 , 4000000 4000000 . 4000000 4000000 . 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000 4000000 . 4000000'
                   'Evaluation Form (scores ONLY): '
                   '- Fluency: 0',
            'Ref': 'Source Code: Code public zip entry ( string name ) { objects require non null ( name , <string> ) ; '
                   'if ( name length ( ) > 0xffff ) { throw new illegal argument exception ( <string> ) ; } this name = '
                   'name ; }'
            # 'Reference Summary: creates a new zip entry with the specified name.\n'
                   'Summary: creates a new zip entry with the name'
                   'Evaluation Form (scores ONLY): '
                   '- Relevance: 4'
                   'Source Code: public void remove scanning callback ( one sheeld scanning callback scanning callback ) { if ( scanning callback != null && scanning callbacks contains ( scanning callback ) ) scanning callbacks remove ( scanning callback ) ; }'
            # 'Reference Summary:  remove a scanning callback .'
                   'Summary: if a scanning and .'
                   'Evaluation Form (scores ONLY): '
                   '- Relevance: 1'
                   'Source Code: private void check server response code ( httpurlconnection url connection ) throws request failure exception { try { if ( url connection get response code ( ) != <num> ) { throw new request failure exception ( <string> + url connection get response code ( ) + <string> ) ; } } catch ( ioexception e ) { throw new request failure exception ( <string> , e ) ; } }'
            # 'Reference Summary: confirms that the omaha server sent back an " ok " code .'
                   'Summary: code that code in the code response .'
                   'Evaluation Form (scores ONLY): '
                   '- Relevance: 0'
                   'Source Code: private void offset start time if necessary ( time start time , time end time , string rrule , calendar event model model ) { if ( rrule == null || rrule is empty ( ) ) { return ; } m event recurrence parse ( rrule ) ; if ( m event recurrence freq != event recurrence weekly ) { return ; } if ( m event recurrence byday length > m event recurrence byday count ) { return ; } int closest weekday = integer max value ; int weekstart = event recurrence day2time day ( m event recurrence wkst ) ; int start day = start time week day ; for ( int i = <num> ; i < m event recurrence byday count ; i ++ ) { int day = event recurrence day2time day ( m event recurrence byday [ i ] ) ; if ( day == start day ) { return ; } if ( day < weekstart ) { day += <num> ; } if ( day > start day && ( day < closest weekday || closest weekday < start day ) ) { closest weekday = day ; } if ( closest weekday == integer max value || closest weekday < start day ) { if ( day < closest weekday ) { closest weekday = day ; } } } if ( closest weekday < start day ) { closest weekday += <num> ; } int days offset = closest weekday - start day ; start time month day += days offset ; end time month day += days offset ; long new start time = start time normalize ( true ) ; long new end time = end time normalize ( true ) ; model m start = new start time ; model m end = new end time ; }'
            # 'Reference Summary:  if the recurrence rule is such that the event start date doesn \' t actually fall in one of the recurrences , then push the start date up to the first actual instance of the event .'
                   'Summary:  parses the elements and , the store . the .'
                   'Evaluation Form (scores ONLY): '
                   '- Relevance: 0'
                   'Source Code: public dexportprivatekeyopenssl ( jframe parent , string entry alias , password quality config password quality config ) { super ( parent , dialog modality type document modal ) ; this entry alias = entry alias ; this password quality config = password quality config ; init components ( ) ; }'
            # 'Reference Summary: creates a new dexportprivatekeyopenssl dialog .'
                   'Summary:  creates a new dexportprivatekeypvk dialog .'
                   'Evaluation Form (scores ONLY): '
                   '- Relevance: 0'
                   'Source Code: override public void on success ( dlsn value ) { if ( value get log segment sequence no ( ) != current log segment seq no ) { log error ( <string> , value get log segment sequence no ( ) , current log segment seq no ) ; errors found set ( true ) ; } if ( verify entry id && value get entry id ( ) != current entry id ) { log error ( <string> , value get entry id ( ) , current entry id ) ; errors found set ( true ) ; } sync latch count down ( ) ; }'
            # 'Reference Summary:  invoked if the computation completes successfully'
                   'Summary: invoked completes a computation successfully successfully'
                   'Evaluation Form (scores ONLY): '
                   '- Relevance: 4'
                   'Source Code: public static void modify file ( file file , function < string , string > modifier ) throws ioexception { string content = new string ( files to byte array ( file ) , standard charsets utf 8 ) ; string result = modifier apply ( content ) ; files write ( result get bytes ( standard charsets utf 8 ) , file ) ; }'
            # 'Reference Summary: modifies the given file in place .'
                   'Summary: modifies the the file given . for prefixes . the file'
                   'Evaluation Form (scores ONLY): '
                   '- Relevance: 2'
                   'Source Code: public process execute async ( final command line command , map < string , string > environment ) throws ioexception { if ( working directory != null && ! working directory exists ( ) ) { throw new ioexception ( working directory + <string> ) ; } return execute internal ( command , environment , working directory , stream handler , null ) ; }'
            # 'Reference Summary: methods for starting asynchronous execution .'
                   'Summary: methods for starting asynchronous execution . process process process process parent parent'
                   'Evaluation Form (scores ONLY): '
                   '- Relevance: 3',
        }

    df = pd.read_excel('../../dataset/RQ1-2/final/recode.xlsx')

    # Define the columns for the results DataFrame
    columns = ['Code', 'Target', 'Generated']

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=columns)

    for idx, row in df.iterrows():
        code_to_display = row['Code']
        target = row['Target']
        generated = row['Generated']
        print(idx)
        print(f"Code: {code_to_display}")
        print(f"Reference: {target}")
        print(f"Summary (To Be Evaluated): {generated}")
        scores_dict = {
            'Code': code_to_display,
            'Target': target,
            'Generated': generated
        }


        for (role_name, role_description), (criterion_name, criterion_task), (eval_name, eval_step), \
            (example_name, example_data) in zip(roles.items(), criteria.items(), evaluation_step.items(),
                                                example.items()):
            if cot:
                prompt = f"""
                {role_description}
                You will be given one summary written for a source code. 
                Your task is to rate the summary on one metric.
                Please make sure you read and understand these instructions carefully. 
                Please keep this document open while reviewing, and refer to it as needed.
                Evaluation Criteria:
                {criterion_name}(0-4) - {criterion_task}
                Evaluation Steps:
                {eval_step}
                Example:
                {example_data}
                Evaluate item:
                Source Code: {code_to_display}
                Reference Summary: {target}
                Summary: {generated}
                {rating}
                """
            else:
                prompt = f"""
                {role_description}
                You will be given one summary written for a source code. 
                Your task is to rate the summary on one metric.
                Please make sure you read and understand these instructions carefully. 
                Please keep this document open while reviewing, and refer to it as needed.
                Evaluation Criteria:
                {criterion_name}(0-4) - {criterion_task}
                Example:
                {example_data}
                Evaluate item:
                Source Code: {code_to_display}
                Reference Summary: {target}
                Summary: {generated}
                {rating}
                """
            score = model_api(model, prompt)
            # print(prompt)
            column_name = f"{role_name} ({criterion_name} Score)"

            if rating_form:
                match = re.search(r'Rating:\s*(\d+\.?\d*)', score)
                if match:
                     match = float(match.group(1))
                else:
                     match = 0
            else:
                match = re.search(r'\d+', score)
                if match:
                     match = match.group()
                else:
                     match = 0

            scores_dict[column_name] = match
            # Printing out the desired information:
            print(f"Role: {role_name}")
            print(f"Criterion: {criterion_name}")
            print(f"Score: {match}")
        print("------" * 10)
        # Append the result to the DataFrame
        results_df = results_df.append(scores_dict, ignore_index=True)

    # Save the results to an Excel file
    results_df.to_excel(f"evaluated_by_{model}_cot{cot}_rating{rating_form}_reference{reference}.xlsx", index=False)


def model_api(model, prompt):
    # print(f"new prompt:\n {prompt}")
    if model == 'gpt-4' or model == 'gpt-3.5-turbo-0613':
        message = [
            {"role": "user", "content": prompt}
        ]
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=message,
            )
            generated_answer = ' '.join(response.choices[0]['message']['content'].strip().split())
        except Exception as e:
            time.sleep(25)
            return model_api(model, prompt)
    else:
        try:
            response = openai.Completion.create(
                engine=model,  # gpt-4, gpt-3.5-turbo, text-davinci-003, text-davinci-002
                prompt=prompt,
                max_tokens=100,
            )
            generated_answer = ' '.join(response.choices[0].text.strip().split())
        except Exception as e:
            time.sleep(25)
            return model_api(model, prompt)
    return generated_answer


if __name__ == '__main__':
    model = "text-davinci-003"
    # model = 'gpt-3.5-turbo-0613'
    # model = 'gpt-4'
    cot = 0  # 0-false, 1-ture
    rating_form = 2  # 0-raw, 1-analyse, 2-explain
    reference = 1  # 0-false, 1-ture
    print(rating_form)
    evaluate( model, cot, rating_form, reference)

