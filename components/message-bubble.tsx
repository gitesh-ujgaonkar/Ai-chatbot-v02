import { cn } from "@/lib/utils"

interface MessageBubbleProps {
  message: string
  isUser: boolean
}

export default function MessageBubble({ message, isUser }: MessageBubbleProps) {
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[80%] rounded-lg px-4 py-2",
          isUser
            ? "bg-primary text-primary-foreground rounded-br-none"
            : "bg-muted text-muted-foreground rounded-bl-none",
        )}
      >
        <p className="whitespace-pre-wrap break-words">{message}</p>
      </div>
    </div>
  )
}
