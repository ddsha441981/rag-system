import re
from typing import Dict, List, Optional
from collections import defaultdict


class TechnicalAccuracyValidator:

    def __init__(self):

        self.technical_knowledge_base = {
            'java_threading': {
                'yield()': {
                    'correct_behavior': [
                        'cooperative scheduling hint',
                        'voluntarily pause current thread',
                        'allows other threads to run',
                        'no guarantee of thread switching',
                        'scheduler dependent',
                        'non-blocking operation'
                    ],
                    'incorrect_patterns': [
                        'blocks until completion',
                        'waits for other threads',
                        'synchronization mechanism',
                        'main thread completes first',
                        'guarantees execution order'
                    ],
                    'type': 'scheduling'
                },

                'join()': {
                    'correct_behavior': [
                        'blocks calling thread',
                        'waits for target thread completion',
                        'synchronization mechanism',
                        'ensures thread finishes',
                        'thread coordination',
                        'blocking operation'
                    ],
                    'incorrect_patterns': [
                        'cooperative scheduling',
                        'main thread completes before others',
                        'non-blocking hint',
                        'voluntary pause',
                        'scheduler hint'
                    ],
                    'type': 'synchronization'
                },

                'wait()': {
                    'correct_behavior': [
                        'object monitor wait',
                        'conditional waiting',
                        'releases monitor lock',
                        'waits for notification',
                        'must be in synchronized context'
                    ],
                    'incorrect_patterns': [
                        'thread scheduling',
                        'cooperative yielding',
                        'non-blocking operation'
                    ],
                    'type': 'synchronization'
                },

                'sleep()': {
                    'correct_behavior': [
                        'timed pause',
                        'thread suspension',
                        'does not release locks',
                        'static method',
                        'interruption possible'
                    ],
                    'incorrect_patterns': [
                        'cooperative scheduling',
                        'releases monitor locks',
                        'synchronization primitive'
                    ],
                    'type': 'timing'
                }
            },

            'synchronization_concepts': {
                'blocking': {
                    'correct_behavior': [
                        'thread waits for condition',
                        'execution suspended',
                        'cannot proceed until condition met'
                    ],
                    'incorrect_patterns': [
                        'cooperative yielding',
                        'voluntary pause',
                        'scheduler hint'
                    ]
                },

                'non-blocking': {
                    'correct_behavior': [
                        'does not wait',
                        'returns immediately',
                        'may fail if resource unavailable'
                    ],
                    'incorrect_patterns': [
                        'waits for completion',
                        'blocks until condition met'
                    ]
                }
            }
        }


        self.common_misconceptions = [
            {
                'pattern': r'main thread.*complete.*before',
                'correction': 'join() makes the calling thread wait for the target thread to complete',
                'severity': 'high'
            },
            {
                'pattern': r'yield.*synchronization',
                'correction': 'yield() is for cooperative scheduling, not synchronization',
                'severity': 'high'
            },
            {
                'pattern': r'join.*scheduling.*hint',
                'correction': 'join() is a synchronization mechanism, not a scheduling hint',
                'severity': 'high'
            }
        ]

        print("✅ Technical Accuracy Validator initialized")

    def validate_technical_response(self, query: str, response: str,
                                    query_classification: Dict, context: List[Dict]) -> Dict:

        validation_result = {
            'is_accurate': True,
            'confidence_score': 1.0,
            'issues_found': [],
            'corrections_suggested': [],
            'technical_score': 1.0,
            'validation_details': {}
        }

        if query_classification.get('type') != 'technical':
            return validation_result


        query_methods = query_classification.get('methods_mentioned', [])
        response_methods = self._extract_methods_from_text(response)


        for method in set(query_methods + response_methods):
            method_validation = self._validate_method_explanation(method, response, query)

            if not method_validation['accurate']:
                validation_result['is_accurate'] = False
                validation_result['issues_found'].extend(method_validation['issues'])
                validation_result['corrections_suggested'].extend(method_validation['corrections'])


        misconception_issues = self._check_common_misconceptions(response)
        if misconception_issues:
            validation_result['is_accurate'] = False
            validation_result['issues_found'].extend(misconception_issues)


        if query_classification.get('subtype') == 'method_comparison':
            comparison_validation = self._validate_method_comparison(query_methods, response)

            if not comparison_validation['accurate']:
                validation_result['is_accurate'] = False
                validation_result['issues_found'].extend(comparison_validation['issues'])
                validation_result['corrections_suggested'].extend(comparison_validation['corrections'])


        consistency_check = self._check_technical_consistency(response, context)
        validation_result['technical_score'] = consistency_check['score']


        validation_result['confidence_score'] = self._calculate_confidence_score(validation_result)

        return validation_result

    def _validate_method_explanation(self, method: str, response: str, query: str) -> Dict:


        method_lower = method.lower()
        response_lower = response.lower()


        if method_lower not in self.technical_knowledge_base.get('java_threading', {}):
            return {'accurate': True, 'issues': [], 'corrections': []}

        method_kb = self.technical_knowledge_base['java_threading'][method_lower]

        issues = []
        corrections = []


        for incorrect_pattern in method_kb['incorrect_patterns']:
            if incorrect_pattern.lower() in response_lower:
                issues.append(f"Incorrect statement about {method}: contains '{incorrect_pattern}'")
                corrections.append(f"{method} behavior: {'; '.join(method_kb['correct_behavior'][:2])}")


        correct_mentions = sum(1 for behavior in method_kb['correct_behavior']
                               if any(word in response_lower for word in behavior.lower().split()))

        if correct_mentions == 0:
            issues.append(f"Missing accurate description of {method} behavior")
            corrections.append(f"{method} correct behavior: {method_kb['correct_behavior'][0]}")

        return {
            'accurate': len(issues) == 0,
            'issues': issues,
            'corrections': corrections
        }

    def _validate_method_comparison(self, methods: List[str], response: str) -> Dict:


        if len(methods) < 2:
            return {'accurate': True, 'issues': [], 'corrections': []}

        issues = []
        corrections = []
        response_lower = response.lower()

        method_types = {}
        for method in methods:
            method_lower = method.lower()
            if method_lower in self.technical_knowledge_base.get('java_threading', {}):
                method_types[method] = self.technical_knowledge_base['java_threading'][method_lower]['type']

        if 'yield' in [m.lower() for m in methods] and 'join' in [m.lower() for m in methods]:


            yield_correct = any(term in response_lower for term in
                                ['cooperative', 'scheduling', 'hint', 'voluntary'])
            join_correct = any(term in response_lower for term in
                               ['synchronization', 'blocking', 'wait', 'completion'])

            if not yield_correct:
                issues.append("Missing correct characterization of yield() as cooperative scheduling")
                corrections.append("yield() is a cooperative scheduling hint that voluntarily pauses the thread")

            if not join_correct:
                issues.append("Missing correct characterization of join() as synchronization mechanism")
                corrections.append("join() is a synchronization mechanism that blocks until thread completion")

            if 'join' in response_lower and 'main thread complete' in response_lower:
                issues.append("Incorrect explanation: join() doesn't make main thread complete first")
                corrections.append("join() makes the calling thread wait for the target thread to complete")

        return {
            'accurate': len(issues) == 0,
            'issues': issues,
            'corrections': corrections
        }

    def _check_common_misconceptions(self, response: str) -> List[str]:


        issues = []
        response_lower = response.lower()

        for misconception in self.common_misconceptions:
            if re.search(misconception['pattern'], response_lower):
                issues.append({
                    'issue': f"Common misconception detected: {misconception['pattern']}",
                    'correction': misconception['correction'],
                    'severity': misconception['severity']
                })

        return issues

    def _check_technical_consistency(self, response: str, context: List[Dict]) -> Dict:


        consistency_score = 1.0
        issues = []


        technical_claims = self._extract_technical_claims(response)


        for claim in technical_claims:
            if self._contradicts_context(claim, context):
                consistency_score -= 0.2
                issues.append(f"Potential contradiction with source material: {claim}")

        return {
            'score': max(0.0, consistency_score),
            'issues': issues
        }

    def _extract_methods_from_text(self, text: str) -> List[str]:

        method_pattern = r'\b(\w+)\s*\(\s*\)'
        methods = re.findall(method_pattern, text, re.IGNORECASE)
        return [method.lower() for method in methods]

    def _extract_technical_claims(self, response: str) -> List[str]:

        sentences = re.split(r'[.!?]+', response)
        technical_claims = []

        technical_indicators = [
            'blocks', 'waits', 'synchroniz', 'cooperativ', 'schedul',
            'thread', 'method', 'function', 'complet', 'pause'
        ]

        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in technical_indicators):
                if len(sentence.strip()) > 10:
                    technical_claims.append(sentence.strip())

        return technical_claims

    def _contradicts_context(self, claim: str, context: List[Dict]) -> bool:

        claim_lower = claim.lower()

        for chunk in context:
            chunk_text = chunk.get('text', '').lower()

            if 'yield' in claim_lower and 'join' in claim_lower:
                if 'cooperative' in chunk_text and 'blocking' in chunk_text:

                    if ('yield' in claim_lower and 'blocking' in claim_lower) or \
                            ('join' in claim_lower and 'cooperative' in claim_lower):
                        return True

        return False

    def _calculate_confidence_score(self, validation_result: Dict) -> float:

        if validation_result['is_accurate']:
            return min(1.0, validation_result['technical_score'])


        issue_penalty = len(validation_result['issues_found']) * 0.2
        return max(0.0, validation_result['technical_score'] - issue_penalty)

    def generate_correction_prompt(self, original_query: str, incorrect_response: str,
                                   validation_result: Dict) -> str:


        corrections = validation_result.get('corrections_suggested', [])
        issues = validation_result.get('issues_found', [])

        correction_prompt = f"""
The following technical response contains inaccuracies that need correction:

ORIGINAL QUERY: {original_query}

RESPONSE WITH ISSUES: {incorrect_response}

SPECIFIC TECHNICAL ISSUES IDENTIFIED:
{chr(10).join([f"- {issue}" for issue in issues])}

REQUIRED CORRECTIONS:
{chr(10).join([f"- {correction}" for correction in corrections])}

Please provide a corrected response that:
1. Addresses each technical issue identified above
2. Uses precise technical terminology
3. Clearly explains the actual behavior of each method/concept
4. Provides accurate comparisons if this is a comparison query
5. Cites the technical documentation accurately

CORRECTED TECHNICAL RESPONSE:"""

        return correction_prompt
