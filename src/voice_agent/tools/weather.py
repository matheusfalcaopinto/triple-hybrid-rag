"""
Example Weather Tool for MCP

This is a demonstration tool showing how to add custom functionality.
Replace with real weather API integration (OpenWeatherMap, WeatherAPI, etc.)
"""

def get_weather(location: str) -> dict:
    """
    Get current weather for a location.
    
    NOTE: This is mock data. Integrate with a real weather API:
    - OpenWeatherMap: https://openweathermap.org/api
    - WeatherAPI: https://www.weatherapi.com/
    - WTTR.in: https://github.com/chubin/wttr.in
    
    Args:
        location: City name or location
        
    Returns:
        Weather information dictionary
    """
    mock_weather_data = {
        "São Paulo": {"temp": "22°C", "condition": "Parcialmente nublado", "humidity": "65%"},
        "Rio de Janeiro": {"temp": "28°C", "condition": "Ensolarado", "humidity": "70%"},
        "Brasília": {"temp": "24°C", "condition": "Céu limpo", "humidity": "45%"},
        "Salvador": {"temp": "30°C", "condition": "Ensolarado", "humidity": "75%"},
        "Fortaleza": {"temp": "31°C", "condition": "Ensolarado", "humidity": "80%"},
    }
    
    weather = mock_weather_data.get(location)
    
    if weather:
        return {
            "location": location,
            "temperature": weather["temp"],
            "condition": weather["condition"],
            "humidity": weather["humidity"],
            "note": "⚠️ This is mock data. Please integrate with a real weather API."
        }
    else:
        return {
            "location": location,
            "temperature": "23°C",
            "condition": "Desconhecido",
            "note": f"⚠️ Mock data for {location}. Please integrate with a real weather API."
        }


# Tool definition - MCP server will automatically discover this
TOOL_DEFINITION = {
    "name": "get_weather",
    "description": "Get current weather information for a Brazilian city (São Paulo, Rio, etc.)",
    "parameters": {
        "location": {
            "type": "string",
            "description": "City name in Brazil (e.g., 'São Paulo', 'Rio de Janeiro', 'Brasília')"
        }
    },
    "required": ["location"],
    "handler": get_weather
}
