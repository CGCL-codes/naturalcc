#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content Type Category Mapper
Categorize content types into several major categories
"""

import pandas as pd
from pathlib import Path
def create_content_category_mapping():
    """Create mapping rules from content types to categories
    """
    
    # Define main categories - reorganize based on functionality and logic
    categories = {
    "User Identity & Personalization": [
        "Profile", "Profile Management", "Settings", "Resume", "CoverLetter",
        "Certificate", "Achievements", "Achievement", "Leaderboard",
        "User Persona", "JourneyMap", "Experience Map", "Attendance",
        "Student Management", "History"
    ],
    "Marketing & Landing Content": [
        "LandingPage", "Landing Page", "Blog", "BlogPost", "Blog/Article", "Article",
        "News Article", "Newsletter", "Email Newsletter", "Portfolio",
        "Testimonials", "Testimonial", "Testimonial/Feedback", "Review",
        "Reviews", "Feedback/Review", "Feedback/Rating", "Rating/Review",
        "Promotions", "Advertisement", "Case Studies", "Informational", "Information"
    ],
    "E-Commerce & Transactions": [
        "Ecommerce", "Market", "Market Overview", "Trading", "Currency Exchange",
        "Checkout", "Payment", "Billing", "Wallet", "Connect Wallet", "My Cards",
        "Card Management", "Invoice", "Order History", "Order Details",
        "OrderDetails", "Order Management", "Order Confirmation", "Order Tracking",
        "OrderTracking", "Delivery", "Delivery Tracking", "Delivery Status",
        "Subscription", "Subscription Management", "Transactions", "Transaction",
        "Transaction History", "TransactionHistory", "TransactionReport",
        "Payment Confirmation", "Rewards", "Referral Program", "Receipts"
    ],
    "Data & Information Presentation": [
        "Dashboard", "List/Feed", "Feed", "Detail", "Report", "Research/Report",
        "Research Documentation", "Comparison/Analysis", "Tracking", "Live Tracking",
        "Progress Tracking", "Task Management", "Management", "Server List",
        "Order History", "Metadata", "Market Overview", "Analysis", "Research"
    ],
    "Forms & Interaction Flows": [
        "Form/Input", "Filter", "Modal/Dialog", "Navigation/TabBar", "Navigation",
        "Navigation/Search", "Search", "Date Picker", "DatePicker", "Upload",
        "Photo Upload", "Download", "Contacts", "Contact", "ContactInfo",
        "File Management", "Component Documentation"
    ],
    "Communication & Collaboration": [
        "Messaging", "Chat", "Chat/Conversation", "Live Chat", "Chat Support",
        "Support Chat", "Chat/Support", "Chatbot", "Chatbot Interaction", "Call",
        "Audio Call", "Video Conference", "Dialer", "Community",
        "Group Chat Creation", "Discussion/Feedback", "Email", "EmailTemplate"
    ],
    "Support, Guidance & Onboarding": [
        "Onboarding", "Help Center", "Help/FAQ", "FAQ", "Help/Support", "Support",
        "Support & Helpdesk", "Tutorial/Guide", "Instructional", "Instructional Guide",
        "Guidelines", "Documentation", "Component Documentation", "Customer Support",
        "Learning", "Service Overview", "Tips"
    ],
    "Notifications & States": [
        "Notification/Alert", "Feedback", "Error/EmptyState", "Confirmation",
        "Verification", "Email Verification", "Feedback/Rating", "Feedback/Review"
    ],
    "Media & Entertainment": [
        "Music Player", "Music Streaming", "Music", "Music Library", "Podcast",
        "Gallery", "Media Player", "MediaPlayback", "Audio Player", "Games",
        "LiveStream"
    ],
    "Scheduling & Activities": [
        "Calendar", "Schedule", "Scheduling", "Training Schedule",
        "Appointment Scheduling", "Reservation", "Itinerary", "Trips",
        "Flight Booking", "Daily Planner", "Timesheet", "World Clock",
        "Workout Timer", "Exercise Timer", "Timer", "Event List"
    ],
    "Health & Lifestyle": [
        "Workout", "Workout/Exercise", "Workout Tracker", "WorkoutSummary",
        "Exercise", "Exercise Guide", "Exercise Timer", "Fitness",
        "Fitness Instruction", "Meal Planning", "Meal Planner", "Recipe",
        "Meditation", "Sleep", "Health Information", "Medical Record",
        "Medical Record Analysis"
    ],
    "Others": [
        "Contract", "Contract/Agreement", "License Agreement", "Loan Management",
        "Integrations", "Language Translation", "Charging", "Evaluation/Assessment",
        "Job Listing", "Order Confirmation", "Notes", "Quote", "Links", "Add Friend",
        "Unknown", "Presentation", "Poll", "Audio Settings", "Live Sports Schedule",
        "Sports Standings", "Sports News", "Map", "LocationSelection",
        "JourneyMap", "Guidelines"
    ]
}

    
    # Create a reverse mapping
    content_to_category = {}
    for category, content_types in categories.items():
        for content_type in content_types:
            content_to_category[content_type] = category
    
    return content_to_category

def process_whitelist_csv(input_file: Path, output_file: Path) -> pd.DataFrame:
    """Process the whitelist file and add a category column"""
    
    # Read the CSV file
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)
    
    # Get mapping rules
    content_to_category = create_content_category_mapping()

    df['content_original'] = df['content']
    df['content'] = df['content_original'].map(content_to_category)
    df['content'] = df['content'].fillna('Others')
    
    # Save results
    print(f"Saving results to {output_file}")
    df.to_csv(output_file, index=False)
    return df

def analyze_content_distribution(df: pd.DataFrame) -> None:
    """Analyze the distribution of content types"""
    
    print("\n=== Content Type Distribution Analysis ===")
    
    # Count original content types
    content_counts = df['content_original'].value_counts()
    print(f"\nTotal number of original content types: {len(content_counts)}")
    
    # Show the top 20 most common types
    print("\nTop 20 most common content types:")
    for i, (content_type, count) in enumerate(content_counts.head(20).items(), 1):
        percentage = (count / len(df)) * 100
        print(f"  {i:2d}. {content_type}: {count} ({percentage:.1f}%)")
    
    # Statistics of distribution after classification
    print("\nDistribution after classification:")
    category_counts = df['content'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    from ...configs.paths import OUTPUT_DIR
    input_file = OUTPUT_DIR / "page_filter" / "split_set" / "whitelist_filter.csv"
    output_file = OUTPUT_DIR / "page_filter" / "whitelist_categorized.csv"
    
    try:
        # Process file
        df = process_whitelist_csv(input_file, output_file)
        # Analyze distribution
        analyze_content_distribution(df)
    except FileNotFoundError:
        print(f"Error: File not found {input_file}")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
