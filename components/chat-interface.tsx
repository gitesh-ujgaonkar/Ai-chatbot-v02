"use client"

import type React from "react"

export default function ChatInterface() {
  return (
    <div className="flex flex-col w-full max-w-3xl mx-auto h-screen">
      {/* Header */}
      <header className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 py-4 px-6 sticky top-0 z-10">
        <h1 className="text-xl font-semibold text-gray-800 dark:text-gray-100">AI Chat Assistant</h1>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-hidden">
        <iframe
          src="https://huggingface.co/spaces/GiteshUjgaonkar/chatbot-v02"
          width="100%"
          height="100%"
          frameBorder="0"
          className="w-full h-full"
        />
      </div>
    </div>
  )
}
