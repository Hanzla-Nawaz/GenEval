import openai
import logging

openai.api_key = 'your-openai-api-key'

logger = logging.getLogger(__name__)

# Function to interact with the chatbot using OpenAI GPT-3.5
def chat_with_bot(prompt, history=[]):
    try:
        # Prepare the conversation history for OpenAI API
        conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
        for message in history:
            if message['role'] == 'user':
                conversation_history.append({"role": "user", "content": message['content']})
            else:
                conversation_history.append({"role": "assistant", "content": message['content']})

        conversation_history.append({"role": "user", "content": prompt})

        # Generate a response from the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.9,
        )

        # Get the response content
        bot_message = response['choices'][0]['message']['content']

        # Append user and bot messages to history
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": bot_message})

        return bot_message, history

    except Exception as e:
        logger.error(f"Error during chat: {e}")
        return "An error occurred while chatting with the bot.", history
