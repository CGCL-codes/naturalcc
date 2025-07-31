import openai
import pandas as pd
import time
import re
import numpy as np
import sys
sys.path.append("../..")
import os
import prettytable as pt
from scipy import stats
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
# openai.api_key = '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'
# openai.api_key = 'sk-QsUMOEHG9tuCYo2PQBjFXOgKXTKEa9eVcOV0JEZR'
openai.api_key = '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'


def evaluate(refs, preds, num, model, reference, turn, approach):
    # 构建角色和评估标准

    criteria = {
        "Coherence": "the summary should be well-structured and well-organized. The summary should not just be a heap "
                     "of related information, but should build from sentence to sentence to a coherent body of "
                     "information about a topic.",
        # Rate the coherence of the summary on a scale of 0-4:\n "
        # "0 - Completely incoherent, lacks logical flow.\n"
        # "1 - Some coherent elements, but significant disconnection in logic.\n"
        # "2 - Generally coherent with occasional breaks in logical flow.\n"
        # "3 - Highly coherent with minor inconsistencies.\n"
        # "4 - Exceptionally structured with clear and seamless logic flow.",

        "Consistency": "the factual alignment between the summary and the summarized code. A factually consistent "
                       "summary contains only statements that are entailed by the source code. Annotators were "
                       "also asked to penalize summaries that contained hallucinated facts. ",
        # "Rate the consistency of the summary on a scale of 0-4:\n"
        # "0 - Numerous inconsistencies and deviations.\n"
        # "1 - Some alignment but with clear discrepancies.\n"
        # "2 - Mostly consistent with minor omissions or deviations.\n"
        # "3 - Highly consistent with very few errors.\n"
        # "4 - Perfectly consistent, accurately reflecting code's intentions and operations.",

        "Fluency": "the quality of individual sentences. The sentence should have no repetitive word, formatting "
                   "problems, capitalization errors or obviously ungrammatical sentences ( "
                   "e.g., fragments, missing components) that make the text difficult to understand.",
        # "Rate the fluency of the summary on a scale of 0,2,4:\n"
        # "0 - with errors and meaningless.\n"
        # "2 - with errors that don't hinder overall understanding.\n"
        # "4 - fluency, ensuring utmost clarity and understanding.",

        "Relevance": "selection of important content from the source. The summary should include only important "
                     "information from the source document. Annotators were instructed to penalize summaries that "
                     "contained redundancies and excess information.",
        # "Rate the relevance of the summary on a scale of 0-4:\n"
        # "0 - Irrelevant details dominate, missing core functionalities.\n"
        # "1 - Some relevant details, but diluted with extraneous information.\n"
        # "2 - Mostly relevant with occasional unnecessary details.\n"
        # "3 - Highly relevant, capturing most pivotal functionalities.\n"
        # "4 - Perfectly relevant, focusing solely on crucial behaviors and actions.",

        # "Comprehensive": "Analyze the summary considering all dimensions - coherence, fluency, consistency, "
        #                  "and relevance. Does it strike a balance in all these areas, representing the code "
        #                  "comprehensively and logically? "
        # "Rate the summary on a scale of 0-4:\n"
        # "0 - Fails in multiple criteria, significant improvements needed.\n"
        # "1 - Meets some criteria, falls short in others.\n"
        # "2 - Adequate in all dimensions but room for improvement.\n"
        # "3 - Strong representation in almost all areas with minor lapses.\n"
        # "4 - A masterful summary, balancing all criteria excellently."
        # "Semantic": 'the semantic similarity between the generated and reference summaries. '
        #             '0 - No similarity between the generation and reference.'
        #             '1 - Have few shared tokens, not semantically similar.'
        #             '2 - Have some shared tokens, probable semantically similar.'
        #             '3 - Much similar in semantic but a few tokens are different.'
        #             '4 - Identical in semantic.'
    }

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
        evaluation_step = {
            'Coh': '',
            'Con': '',
            'Flu': '',
            'Ref': ''
        }
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
        rating = {
            'Coh': 'Evaluation Form (scores ONLY):',
            # 'Con': 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)',
            # 'Flu': 'Evaluation Form (scores ONLY):',
            # 'Ref': 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)',
            'Con': 'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')',
            'Flu': 'Evaluation Form (scores ONLY):',
            'Ref': 'Evaluation Form (scores ONLY):',
        }
    else:
        roles = {
            # coherence
            "Original Code Author 0": "As the Original Code Author, having written the code, you ensure the coherence of the code summary, ensuring that it clearly conveys the main logic "
                                      "of the code and is easy to follow.",
            # Consistency
            "Original Code Author 1": "As the Original Code Author, having written the code, you guarantee that the summary remains consistent with the original code, without hallucinated or unsupported content, similar to fact-checking to prevent any fabricated functionality.",

            # Fluency
            "Original Code Author 2": "As the Original Code Author, having written the code, you focus on ensuring that the summary is written smoothly, with clear sentences and appropriate "
                                      "wording, ensuring it reads naturally, like it was written by a fluent native speaker.",

            # Relevance
            "Code Reviewer": "As a Code Reviewer, serving as an experienced developer, you identify and preserve the most important parts of the code, avoiding unnecessary or off-topic content—like aiming at the core message without distraction.",
        }
        evaluation_step = {
            'Coh': '',
            # 'Evaluation Steps:'
            # '1. Read the source code carefully and understand its main functionality and key operations.'
            # '2. Read the code comments and compare them to the source code. Check if the comments accurately describe'
            # 'the main functionality and key operations of the code, and if they present them in a clear and '
            # 'logical order. '
            # '3. Assign a score for coherence on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
            'Con': '',
            # 'Evaluation Steps:'
            # '1. Read the Source Code carefully and understand its main functionality and any key operations.'
            # '2. Read the code comments and compare them to the source code to evaluate its factual alignment.'
            # 'Ensure that the summary contains only statements that are present or implied in the source code.'
            # 'Be on the lookout for any hallucinated facts or information in the summary that isn\'t supported by the'
            # 'source code. If any are found, they should be penalized in your evaluation.'
            # '3. Assign a score for consistency on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
            'Flu': '',
                   # 'Evaluation Steps:'
                   # '1. Read the code comments carefully and examine each sentence to ensure it is grammatically correct.'
                   # '2. Identify any glaring grammatical errors, such as sentence fragments, missing components like verbs or subjects, or any other issue that makes the text difficult to understand '
                   # '3. Check for any instances of repetitive words that can hamper clarity and ensure proper capitalization throughout the comments.'
                   # '4. Assign a score for fluency on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
                   # 'based on the Evaluation Criteria. ',
            'Ref': '',
            # 'Evaluation Steps:'
            # '1. Read the source code carefully and understand its key information and primary actions of the code.'
            # '2. Read the code comments and compare them to the source code. '
            # 'Evaluate the completeness of the main information. The summary should provide a complete explanation of the main information without omitting significant details.'
            # '3. Check if the code comments include repetitive or unnecessary information. '
            # 'Annotators should be vigilant about penalizing summaries that deviate from the source code\'s primary intent by including tangential or redundant data.'
            # '4. Assign a score for reference on a scale of 0 to 4, where 0 is the lowest and 4 is the highest, '
            # 'based on the Evaluation Criteria. ',
        }
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
        rating = {
            'Coh': 'Evaluation Form (scores ONLY):',
            # 'Con': 'Evaluation Form (Answer by starting with ``Rating:'' and then give the explanation of the rating on the next line by ``Rationale:'')',
            # 'Flu': 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)',
            # 'Ref': 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)',
            'Con': 'Evaluation Form (scores ONLY):',
            'Flu': 'Evaluation Form (scores ONLY):',
            'Ref': 'Evaluation Form (Answer by starting with ``Analysis:'' to analyze the given example regarding the evaluation criteria as concisely as possible, and then give the numeric rating on the next line by ``Rating'':)',
        }
    # 结合数据
    df = pd.DataFrame(preds, columns=['Generated'])
    df = pd.concat([refs, df], axis=1)
    df = df.head(num)
    # Define the columns for the results DataFrame
    columns = ['Code', 'Target', 'Generated']

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame(columns=columns)

    for idx, row in df.iterrows():
        code_to_display = row['Code']
        target = row['Target']
        generated = row['Generated']
        # print(idx)
        # print(f"Code: {code_to_display}")
        # print(f"Reference: {target}")
        # print(f"Summary (To Be Evaluated): {generated}")
        scores_dict = {
            'Code': code_to_display,
            'Target': target,
            'Generated': generated
        }

        for (role_name, role_description), (criterion_name, criterion_task), (eval_name, eval_step), \
            (example_name, example_data),(rating_name, rating_data) in zip(roles.items(), criteria.items(), evaluation_step.items(),
                                                example.items(), rating.items()):

            prompt = f"""
            {role_description}
            You will be given one summary written for a source code. 
            Your task is to rate the summary on one metric.
            Please make sure you read and understand these instructions carefully. 
            Please keep this document open while reviewing, and refer to it as needed.
            Evaluation Criteria:
            {criterion_name}(0-4) - {criterion_task}
            {eval_step}
            Evaluate item:
            Source Code: {code_to_display}
            Reference Summary: {target}
            Summary: {generated}
            {rating_data}
            """
            score = model_api(model, prompt)
            # print(prompt)
            column_name = f"{role_name} ({criterion_name} Score)"
            if reference:
                if rating_name in ['Con']:
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
            else:
                if rating_name in ['Ref']:
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
            # print(f"Role: {role_name}")
            # print(f"Criterion: {criterion_name}")
            # print(f"Score: {score}")
        # print("------" * 10)
        # Append the result to the DataFrame
        results_df = results_df.append(scores_dict, ignore_index=True)
    # 选择最后四列
    last_four_columns = results_df.iloc[:, -1:]
    name = results_df.columns[-1:]
    print(name)
    # 计算这四列每行的平均值
    row_averages = last_four_columns.mean(axis=1)/0.04
    # 将平均值保存到新列 "Average"
    # results_df['Average'] = row_averages
    # Save the results to an Excel file
    results_df.to_excel(f"evaluated_0shot_{approach}_{num}_by_{model}_reference{reference}_turn{turn}.xlsx", index=False)
    # score = results_df[['Average']].mean() / 4
    # print(f'Final evaluated score:{score}')
    return row_averages

def get_bleu_cn(refs,preds):
    r_str_list = []
    p_str_list = []
    for r, p in zip(refs, preds):
        if len(r[0]) == 0 or len(p) == 0:
            continue
        r_str_list.append([" ".join([str(token_id) for token_id in r[0]])])
        p_str_list.append(" ".join([str(token_id) for token_id in p]))
    try:
        bleu_list = codenn_smooth_bleu(r_str_list, p_str_list)
    except:
        bleu_list = [0, 0, 0, 0]
    codenn_bleu = bleu_list[0]
    codenn_bleu = round(codenn_bleu,4)
    return codenn_bleu
def get_bleu_dc(refs,preds):
    all_score = 0.0
    count = 0
    for r, p in zip(refs, preds):
        # nltk bug: https://github.com/nltk/nltk/issues/2204
        if len(p) == 1:
            continue
        score = nltk.translate.bleu(r, p, smoothing_function=SmoothingFunction().method4)
        all_score += score
        count += 1
    emse_bleu = round(all_score / count * 100, 4)
    return emse_bleu
def get_codenn_score(refs, preds):
    r_str_list = []
    p_str_list = []
    for r, p in zip(refs, preds):
        if len(r[0]) == 0 or len(p) == 0:
            continue
        r_str_list.append([" ".join([str(token_id) for token_id in r[0]])])
        p_str_list.append(" ".join([str(token_id) for token_id in p]))
    try:
        bleu_list = codenn_smooth_bleu(r_str_list, p_str_list)
    except:
        bleu_list = [0, 0, 0, 0]
    codenn_bleu = bleu_list[0]
    codenn_bleu = round(codenn_bleu,4)
    return codenn_bleu
def get_score(r0, r1):
#     pvalue = wilcoxon_signed_rank_test(y1, y2)
    _, p_val_t_test = stats.ttest_ind(r0, r1, equal_var=False)
    _, p_val_wwu_test = stats.mannwhitneyu(r0, r1, alternative='two-sided')
    return p_val_t_test, p_val_wwu_test

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
    elif model == 'gpt-4.1-nano' or model == 'gpt-4o-mini' or model == 'gpt-4o':
        try:
            message = [
                      {"role": "user", "content": prompt}
               ]
            response = openai.chat.completions.create(
              model=model,
              messages=message,
              response_format={
                "type": "text"
              },
              temperature=1,
              max_completion_tokens=1000,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0,
              store=False
            )
            generated_answer = ' '.join(response.choices[0].message.content.strip().split())
            matches = re.findall(r'\d+', generated_answer)
            match = matches[-1]  # 获取最后一个匹配的数字
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

def get_mean_std(arr):
    arr_mean = np.mean(arr)
    arr_std = np.std(arr, ddof=1)
    return arr_mean ,  arr_std

def get_pvalue_and_effect_size(all_score):
    models_name = list(all_score)
    for i in range(len(models_name)):
        for j in range(i + 1, len(models_name)):
            pvalue = get_score(all_score[models_name[i]], all_score[models_name[j]])
            print("{} and {}, t-test:pvalue is {}, WMW-test: pvalue is {}".format(models_name[i], models_name[j], pvalue[0], pvalue[1]))

def read_to_df(filename):
    f = open(filename, 'r',encoding="utf-8")
    res = []
    for row in f:
        # (rid, text) = row.split('\t')
        res.append(row.rstrip('\n'))
    return res

def show_dict(all_bleu):
    tb = pt.PrettyTable()
    tb.field_names = all_bleu.keys()
    tb.add_row(all_bleu.values())
    print(tb)

def get_all_datset_result(approaches, diff_datasets, evaluate, num, model, reference, turn):
    #     scores = {}
    scores_p_test = {}
    for approach in approaches:
        scores_diff_dataset = {}
        latex_diff_dataset = {}
        for diff_dataset in diff_datasets:
            average_score = []
            for random_seed in range(1):
                refs_filename = os.path.join('../../dataset/RQ3/final/' + diff_dataset, "code%s.xlsx" % random_seed)
                # refs_filename = os.path.join('../../dataset/RQ3/final/' + diff_dataset, approach, "random%s" % random_seed, "test.pred")
                # if diff_dataset in ["FCM", "FCM_project_large"]:
                #     refs_filename = os.path.join(diff_dataset, approach, "code.xlsx")
                preds_filename = os.path.join('../../dataset/RQ3/final/' + diff_dataset, approach, "random%s" % random_seed, "results.xlsx")
                preds = pd.read_excel(preds_filename)
                refs = pd.read_excel(refs_filename)
                bleu_s = evaluate(refs, preds, num, model, reference, turn, approach)
                # bleu_s = get_codenn_score(refs, preds)
                average_score.append(bleu_s)
                if diff_dataset in list(scores_p_test):
                    scores_p_test[diff_dataset].append(bleu_s)
                else:
                    scores_p_test[diff_dataset] = [bleu_s]
            mean, std_var = get_mean_std(average_score)
            scores_diff_dataset[diff_dataset] = "%.2f+-%.2f" % (mean, std_var)
            latex_diff_dataset[diff_dataset] = "&%.2f$\pm$%.2f" % (mean, std_var)
        #         scores[diff_dataset] = scores_diff_dataset
        print(15 * "*" + approach + 15 * "*")
        print(approach)
        print(average_score)

        show_dict(scores_diff_dataset)
    print("-------------Significant Testing---------------")
    #     print(scores_p_test)
    if len(scores_p_test) > 1:
        get_pvalue_and_effect_size(dict(list(scores_p_test.items())[:1]))
        get_pvalue_and_effect_size(dict(list(scores_p_test.items())[1:]))
    else:
        get_pvalue_and_effect_size(scores_p_test)

if __name__ == '__main__':
    num = 100
    # model = "text-davinci-003"
    # model = 'gpt-3.5-turbo-0613'
    # model = 'gpt-4'
    model = 'gpt-4o-mini'
    # cot = 1  # 0-false, 1-ture
    # rating_form = 0  # 0-raw, 1-explain, 2-analyse
    reference = 0  # 0-false, 1-ture

    # sys.stdout = open(f'log_evaluated_{num}_by_{model}.txt', 'w')

    # Performance in Different Dtaset: TCL FCM CSN
    diff_datasets = ["TLC"]
    # approaches = ["codenn", "deepcom", "astattgru", "rencos", "ncs"]
    # approaches = ["chatgpt", "codex"]
    approaches = ["codeT5"]
    turn = 1

    get_all_datset_result(approaches, diff_datasets, evaluate, num, model, reference, turn)