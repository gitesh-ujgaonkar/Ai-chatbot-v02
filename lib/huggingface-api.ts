"use server"

export async function sendMessageToAI(message: string): Promise<string> {
  // Get the Hugging Face token from environment variables
  const token = process.env.HUGGINGFACE_API_TOKEN

  if (!token) {
    throw new Error("HUGGINGFACE_API_TOKEN is not defined in environment variables")
  }

  try {
    const response = await fetch("https://api-inference.huggingface.co/models/GiteshUjgaonkar/chatbot-v02", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ inputs: message }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => null)
      console.error("API error:", errorData)
      throw new Error(`API error: ${response.status}`)
    }

    const data = await response.json()

    // Handle different response formats from Hugging Face
    if (Array.isArray(data) && data.length > 0) {
      if (typeof data[0].generated_text === "string") {
        return data[0].generated_text
      }
    }

    // Fallback if the response format is different
    return JSON.stringify(data)
  } catch (error) {
    console.error("Error calling Hugging Face API:", error)
    throw error
  }
}
