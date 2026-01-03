"""
GEO Audit Analyzer: Analyzes content for GEO optimization issues.

Provides detailed breakdown of:
- Missing citations/sources
- Lack of statistics/numbers
- Poor structure (headings, sections)
- Missing FAQ sections
- Missing TL;DR/summary
- Readability issues
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class PageAudit:
    """Audit results for a single page."""
    url: str
    title: str
    word_count: int
    issues: List[Dict]
    scores: Dict[str, float]
    recommendations: List[str]


@dataclass
class AuditReport:
    """Complete audit report for all crawled pages."""
    total_pages: int
    overall_score: float
    pages: List[PageAudit]
    summary: Dict
    top_issues: List[str]
    top_recommendations: List[str]


def analyze_citations(content: str) -> Tuple[float, List[str], List[str]]:
    """
    Check for citations, sources, and references.

    Returns: (score, issues, recommendations)
    """
    issues = []
    recommendations = []

    # Patterns that indicate citations
    citation_patterns = [
        r'according to',
        r'research shows',
        r'study found',
        r'experts say',
        r'data from',
        r'source:',
        r'cited by',
        r'published in',
        r'\[\d+\]',  # [1], [2] style citations
        r'et al\.',
        r'university of',
        r'institute of',
    ]

    citation_count = 0
    for pattern in citation_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        citation_count += len(matches)

    # Check for URLs as sources
    url_pattern = r'https?://[^\s\)\]>]+'
    urls = re.findall(url_pattern, content)
    citation_count += len(urls)

    word_count = len(content.split())
    citations_per_500_words = (citation_count / max(word_count, 1)) * 500

    if citations_per_500_words < 1:
        score = 2.0
        issues.append("No citations or sources found")
        recommendations.append("Add citations from authoritative sources (research papers, industry reports, expert quotes)")
    elif citations_per_500_words < 2:
        score = 5.0
        issues.append("Very few citations (less than 2 per 500 words)")
        recommendations.append("Increase citation density - aim for 2-3 citations per 500 words")
    elif citations_per_500_words < 3:
        score = 7.0
        issues.append("Citation coverage could be improved")
        recommendations.append("Add more diverse sources (mix of studies, experts, and data)")
    else:
        score = 9.0

    return score, issues, recommendations


def analyze_statistics(content: str) -> Tuple[float, List[str], List[str]]:
    """
    Check for statistics, numbers, and data points.

    Returns: (score, issues, recommendations)
    """
    issues = []
    recommendations = []

    # Patterns for statistics
    stat_patterns = [
        r'\d+%',  # percentages
        r'\d+\.\d+%',  # decimal percentages
        r'\$[\d,]+',  # dollar amounts
        r'€[\d,]+',  # euro amounts
        r'\d+ (million|billion|thousand)',
        r'\d+x',  # multipliers
        r'\d+ out of \d+',
        r'\d+/\d+',  # fractions
        r'increased by \d+',
        r'decreased by \d+',
        r'grew \d+',
    ]

    stat_count = 0
    for pattern in stat_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        stat_count += len(matches)

    # Also count standalone significant numbers
    numbers = re.findall(r'\b\d{2,}\b', content)
    stat_count += len(numbers) // 2  # Weight these less

    word_count = len(content.split())
    stats_per_500_words = (stat_count / max(word_count, 1)) * 500

    if stats_per_500_words < 1:
        score = 2.0
        issues.append("No statistics or data points found")
        recommendations.append("Add specific numbers, percentages, and data points to support claims")
    elif stats_per_500_words < 2:
        score = 5.0
        issues.append("Very few statistics (less than 2 per 500 words)")
        recommendations.append("Include more quantifiable data - statistics improve AI citation rates by 41%")
    elif stats_per_500_words < 4:
        score = 7.0
        issues.append("Statistics coverage could be improved")
        recommendations.append("Add more specific metrics and benchmarks")
    else:
        score = 9.0

    return score, issues, recommendations


def analyze_structure(content: str) -> Tuple[float, List[str], List[str]]:
    """
    Check content structure: headings, sections, hierarchy.

    Returns: (score, issues, recommendations)
    """
    issues = []
    recommendations = []

    # Check for headings
    h1_count = len(re.findall(r'^# [^\n]+', content, re.MULTILINE))
    h2_count = len(re.findall(r'^## [^\n]+', content, re.MULTILINE))
    h3_count = len(re.findall(r'^### [^\n]+', content, re.MULTILINE))

    total_headings = h1_count + h2_count + h3_count

    # Check for bullet points
    bullet_count = len(re.findall(r'^[\-\*] ', content, re.MULTILINE))
    numbered_count = len(re.findall(r'^\d+\. ', content, re.MULTILINE))

    # Check for paragraphs
    paragraphs = [p for p in content.split('\n\n') if p.strip() and len(p.split()) > 10]

    word_count = len(content.split())

    score = 5.0

    if total_headings == 0:
        score -= 2
        issues.append("No headings found - content lacks structure")
        recommendations.append("Add H1 and H2 headings to organize content into clear sections")
    elif total_headings < 3:
        score -= 1
        issues.append("Very few headings - structure could be improved")
        recommendations.append("Break content into more sections with descriptive headings")
    else:
        score += 2

    if bullet_count + numbered_count == 0:
        score -= 1
        issues.append("No bullet points or numbered lists")
        recommendations.append("Use bullet points to highlight key information - improves AI parsing")
    elif bullet_count + numbered_count < 3:
        issues.append("Limited use of lists")
        recommendations.append("Add more bullet points for key takeaways and features")
    else:
        score += 1

    # Check section length balance
    if paragraphs:
        avg_paragraph_words = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
        if avg_paragraph_words > 200:
            issues.append("Paragraphs are too long (avg > 200 words)")
            recommendations.append("Break long paragraphs into smaller, focused chunks")
            score -= 1

    score = max(1, min(10, score))
    return score, issues, recommendations


def analyze_faq(content: str) -> Tuple[float, List[str], List[str]]:
    """
    Check for FAQ sections and Q&A content.

    Returns: (score, issues, recommendations)
    """
    issues = []
    recommendations = []

    content_lower = content.lower()

    # Check for FAQ section
    has_faq_section = bool(re.search(r'(faq|frequently asked|common questions)', content_lower))

    # Check for Q&A patterns
    qa_patterns = [
        r'\?[\s\n]+[A-Z]',  # Question followed by answer
        r'Q:\s*[^\n]+\s*A:',  # Q: A: format
        r'### [^\n]*\?',  # Heading as question
        r'## [^\n]*\?',
        r'what is [^\?]+\?',
        r'how (do|does|can|to) [^\?]+\?',
        r'why (do|does|is|are) [^\?]+\?',
        r'when (do|does|should|can) [^\?]+\?',
    ]

    qa_count = 0
    for pattern in qa_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        qa_count += len(matches)

    if not has_faq_section and qa_count < 2:
        score = 2.0
        issues.append("No FAQ section found")
        recommendations.append("Add an FAQ section with 3-5 common questions and clear answers")
    elif qa_count < 3:
        score = 5.0
        issues.append("Limited Q&A content")
        recommendations.append("Expand FAQ with more questions users commonly ask")
    elif qa_count < 5:
        score = 7.0
        issues.append("FAQ section could be more comprehensive")
        recommendations.append("Add more detailed answers to FAQ questions")
    else:
        score = 9.0

    return score, issues, recommendations


def analyze_summary(content: str) -> Tuple[float, List[str], List[str]]:
    """
    Check for TL;DR, summary, or key takeaways section.

    Returns: (score, issues, recommendations)
    """
    issues = []
    recommendations = []

    content_lower = content.lower()

    # Check for summary patterns
    summary_patterns = [
        r'tl;?dr',
        r'summary',
        r'key takeaways',
        r'in (short|brief|summary)',
        r'bottom line',
        r'key points',
        r'highlights',
        r'overview',
        r'at a glance',
        r'quick summary',
    ]

    has_summary = False
    for pattern in summary_patterns:
        if re.search(pattern, content_lower):
            has_summary = True
            break

    # Check for introductory summary (first paragraph being concise)
    first_para = content.split('\n\n')[0] if content else ""
    has_intro_summary = len(first_para.split()) < 100 and len(first_para.split()) > 20

    if not has_summary and not has_intro_summary:
        score = 3.0
        issues.append("No TL;DR or summary section found")
        recommendations.append("Add a TL;DR or 'Key Takeaways' section at the top for quick reference")
    elif not has_summary:
        score = 6.0
        issues.append("Could benefit from explicit summary section")
        recommendations.append("Add a 'Key Takeaways' bullet list for AI-friendly extraction")
    else:
        score = 9.0

    return score, issues, recommendations


def analyze_readability(content: str) -> Tuple[float, List[str], List[str]]:
    """
    Check content readability and clarity.

    Returns: (score, issues, recommendations)
    """
    issues = []
    recommendations = []

    words = content.split()
    sentences = re.split(r'[.!?]+', content)
    sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return 5.0, ["Could not analyze readability"], []

    # Average sentence length
    avg_sentence_length = len(words) / len(sentences)

    # Average word length
    avg_word_length = sum(len(w) for w in words) / max(len(words), 1)

    # Check for complex words (3+ syllables approximation: 7+ chars)
    complex_words = [w for w in words if len(w) > 7]
    complex_ratio = len(complex_words) / max(len(words), 1)

    score = 7.0

    if avg_sentence_length > 25:
        score -= 2
        issues.append(f"Sentences are too long (avg {avg_sentence_length:.0f} words)")
        recommendations.append("Shorten sentences to 15-20 words for better clarity")
    elif avg_sentence_length > 20:
        score -= 1
        issues.append(f"Sentences could be shorter (avg {avg_sentence_length:.0f} words)")

    if complex_ratio > 0.2:
        score -= 1
        issues.append(f"High use of complex words ({complex_ratio*100:.0f}%)")
        recommendations.append("Simplify vocabulary - use common words where possible")

    if avg_word_length > 6:
        score -= 1
        issues.append("Content uses many long/technical words")
        recommendations.append("Balance technical terms with simpler explanations")

    score = max(1, min(10, score))
    return score, issues, recommendations


def audit_content(content: str, url: str, title: str) -> PageAudit:
    """
    Run full audit on a piece of content.

    Returns PageAudit with all analysis results.
    """
    all_issues = []
    all_recommendations = []
    scores = {}

    # Run all analyses
    analyses = [
        ("citations", analyze_citations),
        ("statistics", analyze_statistics),
        ("structure", analyze_structure),
        ("faq", analyze_faq),
        ("summary", analyze_summary),
        ("readability", analyze_readability),
    ]

    for name, analyzer in analyses:
        score, issues, recs = analyzer(content)
        scores[name] = score
        all_issues.extend([{"category": name, "issue": issue} for issue in issues])
        all_recommendations.extend(recs)

    # Deduplicate recommendations
    all_recommendations = list(dict.fromkeys(all_recommendations))

    return PageAudit(
        url=url,
        title=title,
        word_count=len(content.split()),
        issues=all_issues,
        scores=scores,
        recommendations=all_recommendations[:5]  # Top 5 recommendations per page
    )


def generate_audit_report(pages_content: List[Dict]) -> AuditReport:
    """
    Generate complete audit report for multiple pages.

    Args:
        pages_content: List of dicts with 'url', 'title', 'content' keys

    Returns:
        Complete AuditReport
    """
    page_audits = []
    all_issues = []
    all_recommendations = []
    all_scores = {
        "citations": [],
        "statistics": [],
        "structure": [],
        "faq": [],
        "summary": [],
        "readability": [],
    }

    for page in pages_content:
        audit = audit_content(
            content=page.get("content", ""),
            url=page.get("url", ""),
            title=page.get("title", "Untitled")
        )
        page_audits.append(audit)
        all_issues.extend(audit.issues)
        all_recommendations.extend(audit.recommendations)

        for key, score in audit.scores.items():
            all_scores[key].append(score)

    # Calculate average scores
    avg_scores = {
        key: sum(values) / len(values) if values else 0
        for key, values in all_scores.items()
    }

    # Calculate overall score (weighted average)
    weights = {
        "citations": 0.25,
        "statistics": 0.20,
        "structure": 0.15,
        "faq": 0.15,
        "summary": 0.10,
        "readability": 0.15,
    }

    overall_score = sum(
        avg_scores[key] * weights[key]
        for key in weights
    )

    # Count issue frequency
    issue_counts = {}
    for issue in all_issues:
        key = issue["issue"]
        issue_counts[key] = issue_counts.get(key, 0) + 1

    # Top issues sorted by frequency
    top_issues = sorted(issue_counts.keys(), key=lambda x: issue_counts[x], reverse=True)[:5]

    # Deduplicate and prioritize recommendations
    rec_counts = {}
    for rec in all_recommendations:
        rec_counts[rec] = rec_counts.get(rec, 0) + 1
    top_recommendations = sorted(rec_counts.keys(), key=lambda x: rec_counts[x], reverse=True)[:5]

    summary = {
        "total_words": sum(p.word_count for p in page_audits),
        "total_issues": len(all_issues),
        "avg_scores": avg_scores,
        "lowest_scoring_area": min(avg_scores, key=avg_scores.get) if avg_scores else None,
        "highest_scoring_area": max(avg_scores, key=avg_scores.get) if avg_scores else None,
    }

    return AuditReport(
        total_pages=len(page_audits),
        overall_score=round(overall_score, 1),
        pages=[asdict(p) for p in page_audits],
        summary=summary,
        top_issues=top_issues,
        top_recommendations=top_recommendations
    )


if __name__ == "__main__":
    # Test the analyzer
    test_content = """
    # Welcome to Our Company

    We are a leading provider of software solutions.

    ## Our Services

    - Web Development
    - Mobile Apps
    - Cloud Solutions

    Contact us today to learn more about how we can help your business grow.
    """

    report = generate_audit_report([{
        "url": "https://example.com",
        "title": "Test Page",
        "content": test_content
    }])

    print(f"Overall Score: {report.overall_score}/10")
    print(f"Top Issues: {report.top_issues}")
    print(f"Top Recommendations: {report.top_recommendations}")
