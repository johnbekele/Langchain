from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import requests
import json

# API Keys
WEATHER_API_KEY = "b5eea87bf3f58faeaaaca0fa442a5abd"
GEMINI_API_KEY = "AIzaSyByTXGB2hWKn0SYSwNKdJHNW5ybbQ3yLBU"

@tool
def get_weather(city: str) -> str:
   
    """
    Fetches current weather data for a given city.
    Use this tool when you need to get weather information.
    
    Args:
        city: The name of the city to get weather for
    """

    print(f"Fetching weather data for {city}")
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(weather_url)
        response.raise_for_status()
        weather_data = response.json()
        
        formatted_data = {
            "city": weather_data.get("name"),
            "temperature": weather_data["main"]["temp"],
            "feels_like": weather_data["main"]["feels_like"],
            "conditions": weather_data["weather"][0]["description"],
            "humidity": weather_data["main"]["humidity"],
            "wind_speed": weather_data["wind"]["speed"],
            "pressure": weather_data["main"]["pressure"]
        }
        print(formatted_data)
        return json.dumps(formatted_data, indent=2)
    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        return f"Error fetching weather data: {str(e)}"

@tool
def search_clothings(query: str, max_results: int = 10) -> str:
    """
    Searches for clothing items based on a query string.
    Searches in title, description, and category fields.
    Returns up to max_results matching items.
    
    Use this tool when you need to find clothing items. If the first search doesn't 
    return good results, try again with different keywords or broader/narrower terms.
    
    Args:
        query: Search terms (e.g., "comfortable", "cotton", "jacket", "soft")
        max_results: Maximum number of results to return (default: 10)
    
    Returns:
        JSON string with matching products, each containing:
        - id, title, price, description, category, image, rating
    """
    clothing_url = "https://fakestoreapi.com"
    
    try:
        response = requests.get(f"{clothing_url}/products" ,verify=False)
        response.raise_for_status()
        clothing_data = response.json()
        
        query_lower = query.lower()
        matches = []
        
        # Search in title, description, and category
        for item in clothing_data:
            title = item.get("title", "").lower()
            description = item.get("description", "").lower()
            category = item.get("category", "").lower()
            
            # Check if query matches in any field
            if (query_lower in title or 
                query_lower in description or 
                query_lower in category or
                any(word in title or word in description for word in query_lower.split())):
                matches.append({
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "price": item.get("price"),
                    "description": item.get("description"),
                    "category": item.get("category"),
                    "rating": item.get("rating", {}),
                })
                if len(matches) >= max_results:
                    break
        
        if matches:
            print(f"Found {len(matches)} matching items for query: '{query}'")
            return json.dumps(matches, indent=2)
        else:
            return f"No clothing items found matching '{query}'. Try different keywords like 'jacket', 'shirt', 'jewelry', or 'electronics'."
    except Exception as e:
        print(f"Error searching clothing items: {str(e)}")
        return f"Error searching clothing items: {str(e)}"

    

# Create the agent
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)

agent = create_agent(
    model=llm,
    tools=[get_weather, search_clothings],
    system_prompt="""You are a helpful assistant that can provide weather information and search for clothing items.

When searching for clothing:

1. **Understand the user's request**: What type of clothing? What qualities matter (comfortable, soft, warm, etc.)?

2. **Start with a focused search**: Use the search_clothings tool with relevant keywords from the user's request.

3. **Analyze the results**:
   - Check if the returned items match what the user is looking for
   - Look at title, description, and category to judge relevance
   - Consider if items are truly "comfortable" based on materials and descriptions

4. **If results are not good enough, try again**:
   - If no results: try broader keywords (e.g., "jacket" instead of "comfortable winter jacket")
   - If results don't match: try different keywords or synonyms
   - If too many results: refine with more specific terms
   - You can call search_clothings multiple times (up to 3-4 attempts) with different queries

5. **Provide a helpful answer**:
   - Show the best matching items (3-5 max)
   - Explain why each item matches the user's criteria
   - If nothing matches well, explain what you found and suggest alternative searches

For weather questions, use the get_weather tool to fetch real data."""
)

# Test the agent
if __name__ == "__main__":
    print("="*60)
    print("Weather Advisor Agent - LangChain System")
    print("="*60)
    
    # Test with clothing search - agent should try multiple queries if needed
    query = "Find me comfortable clothing items. I want something soft and cozy."
    
    print(f"\n💭 You: {query}\n")
    print("🤖 Agent: ", end="", flush=True)
    
    # Use invoke() with messages in the state format
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    # Print the final response from the agent
    final_message = result["messages"][-1]
    
    # Handle both string and list content formats
    if isinstance(final_message.content, str):
        print(final_message.content)
    elif isinstance(final_message.content, list) and len(final_message.content) > 0:
        # If it's a list, try to get text from first item
        first_item = final_message.content[0]
        if isinstance(first_item, dict) and "text" in first_item:
            print(first_item["text"])
        else:
            print(str(first_item))
    else:
        print(final_message.content)