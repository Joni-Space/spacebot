import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
	api,
	type ChannelInfo,
	type InboundMessageEvent,
	type OutboundMessageEvent,
	type TypingStateEvent,
} from "./api/client";
import { useEventSource } from "./hooks/useEventSource";

const queryClient = new QueryClient({
	defaultOptions: {
		queries: {
			staleTime: 30_000,
			retry: 1,
			refetchOnWindowFocus: true,
		},
	},
});

interface ChatMessage {
	id: string;
	sender: "user" | "bot";
	senderName?: string;
	text: string;
	timestamp: number;
}

interface ChannelLiveState {
	isTyping: boolean;
	messages: ChatMessage[];
	historyLoaded: boolean;
}

const VISIBLE_MESSAGES = 6;
const MAX_MESSAGES = 50;

function formatUptime(seconds: number): string {
	const hours = Math.floor(seconds / 3600);
	const minutes = Math.floor((seconds % 3600) / 60);
	const secs = seconds % 60;
	if (hours > 0) return `${hours}h ${minutes}m`;
	if (minutes > 0) return `${minutes}m ${secs}s`;
	return `${secs}s`;
}

function formatTimeAgo(dateStr: string): string {
	const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
	if (seconds < 60) return "just now";
	if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
	if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
	return `${Math.floor(seconds / 86400)}d ago`;
}

function formatTimestamp(ts: number): string {
	return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function platformIcon(platform: string): string {
	switch (platform) {
		case "discord": return "Discord";
		case "slack": return "Slack";
		case "telegram": return "Telegram";
		case "webhook": return "Webhook";
		case "cron": return "Cron";
		default: return platform;
	}
}

function platformColor(platform: string): string {
	switch (platform) {
		case "discord": return "bg-indigo-500/20 text-indigo-400";
		case "slack": return "bg-green-500/20 text-green-400";
		case "telegram": return "bg-blue-500/20 text-blue-400";
		case "cron": return "bg-amber-500/20 text-amber-400";
		default: return "bg-gray-500/20 text-gray-400";
	}
}

function ChannelCard({
	channel,
	liveState,
}: {
	channel: ChannelInfo;
	liveState: ChannelLiveState | undefined;
}) {
	const isTyping = liveState?.isTyping ?? false;
	const messages = liveState?.messages ?? [];
	const visible = messages.slice(-VISIBLE_MESSAGES);

	return (
		<div className="flex flex-col rounded-lg border border-app-line bg-app-darkBox transition-colors hover:border-app-line/80">
			{/* Header */}
			<div className="flex items-start justify-between p-4 pb-2">
				<div className="min-w-0 flex-1">
					<div className="flex items-center gap-2">
						<h3 className="truncate font-medium text-ink">
							{channel.display_name ?? channel.id}
						</h3>
						{isTyping && (
							<div className="flex items-center gap-1">
								<span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
								<span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent [animation-delay:0.2s]" />
								<span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent [animation-delay:0.4s]" />
							</div>
						)}
					</div>
					<div className="mt-1 flex items-center gap-2">
						<span className={`inline-flex items-center rounded-md px-1.5 py-0.5 text-tiny font-medium ${platformColor(channel.platform)}`}>
							{platformIcon(channel.platform)}
						</span>
						<span className="text-tiny text-ink-faint">
							{formatTimeAgo(channel.last_activity_at)}
						</span>
					</div>
				</div>
				<div className="ml-2 flex-shrink-0">
					<div className={`h-2 w-2 rounded-full ${isTyping ? "bg-accent animate-pulse" : "bg-green-500/60"}`} />
				</div>
			</div>

			{/* Message stream */}
			{visible.length > 0 && (
				<div className="flex flex-col gap-1 border-t border-app-line/50 p-3">
					{messages.length > VISIBLE_MESSAGES && (
						<span className="mb-1 text-tiny text-ink-faint">
							{messages.length - VISIBLE_MESSAGES} earlier messages
						</span>
					)}
					{visible.map((message) => (
						<div key={message.id} className="flex gap-2 text-sm">
							<span className="flex-shrink-0 text-tiny text-ink-faint">
								{formatTimestamp(message.timestamp)}
							</span>
							<span className={`flex-shrink-0 text-tiny font-medium ${
								message.sender === "user" ? "text-accent-faint" : "text-green-400"
							}`}>
								{message.sender === "user" ? (message.senderName ?? "user") : "bot"}
							</span>
							<p className="line-clamp-1 text-sm text-ink-dull">{message.text}</p>
						</div>
					))}
				</div>
			)}
		</div>
	);
}

function Dashboard() {
	const { data: statusData } = useQuery({
		queryKey: ["status"],
		queryFn: api.status,
		refetchInterval: 5000,
	});

	const { data: channelsData, isLoading: channelsLoading } = useQuery({
		queryKey: ["channels"],
		queryFn: api.channels,
		refetchInterval: 10000,
	});

	const [liveStates, setLiveStates] = useState<Record<string, ChannelLiveState>>({});

	// Load conversation history for each channel on first appearance
	const channels = channelsData?.channels ?? [];
	useEffect(() => {
		for (const channel of channels) {
			setLiveStates((prev) => {
				if (prev[channel.id]?.historyLoaded) return prev;

				// Mark as loading to prevent duplicate fetches
				const updated = {
					...prev,
					[channel.id]: {
						...prev[channel.id],
						isTyping: prev[channel.id]?.isTyping ?? false,
						messages: prev[channel.id]?.messages ?? [],
						historyLoaded: true,
					},
				};

				// Fetch history async
				api.channelMessages(channel.id, MAX_MESSAGES).then((data) => {
					const history: ChatMessage[] = data.messages.map((message) => ({
						id: message.id,
						sender: message.role === "user" ? "user" as const : "bot" as const,
						senderName: message.sender_name ?? (message.role === "user" ? message.sender_id ?? undefined : undefined),
						text: message.content,
						timestamp: new Date(message.created_at).getTime(),
					}));

					setLiveStates((current) => {
						const existing = current[channel.id];
						if (!existing) return current;
						// Merge: history first, then any SSE messages that arrived during fetch
						const sseMessages = existing.messages;
						const lastHistoryTs = history.length > 0 ? history[history.length - 1].timestamp : 0;
						const newSseMessages = sseMessages.filter((m) => m.timestamp > lastHistoryTs);
						return {
							...current,
							[channel.id]: {
								...existing,
								messages: [...history, ...newSseMessages].slice(-MAX_MESSAGES),
							},
						};
					});
				}).catch((error) => {
					console.warn(`Failed to load history for ${channel.id}:`, error);
				});

				return updated;
			});
		}
	}, [channels]);

	const pushMessage = useCallback((channelId: string, message: ChatMessage) => {
		setLiveStates((prev) => {
			const existing = prev[channelId] ?? { isTyping: false, messages: [], historyLoaded: false };
			const messages = [...existing.messages, message].slice(-MAX_MESSAGES);
			return { ...prev, [channelId]: { ...existing, messages } };
		});
	}, []);

	const handleInboundMessage = useCallback((data: unknown) => {
		const event = data as InboundMessageEvent;
		pushMessage(event.channel_id, {
			id: `in-${Date.now()}-${Math.random()}`,
			sender: "user",
			senderName: event.sender_id,
			text: event.text,
			timestamp: Date.now(),
		});
		queryClient.invalidateQueries({ queryKey: ["channels"] });
	}, [pushMessage]);

	const handleOutboundMessage = useCallback((data: unknown) => {
		const event = data as OutboundMessageEvent;
		pushMessage(event.channel_id, {
			id: `out-${Date.now()}-${Math.random()}`,
			sender: "bot",
			text: event.text,
			timestamp: Date.now(),
		});
		setLiveStates((prev) => ({
			...prev,
			[event.channel_id]: {
				...prev[event.channel_id],
				isTyping: false,
				messages: prev[event.channel_id]?.messages ?? [],
				historyLoaded: prev[event.channel_id]?.historyLoaded ?? false,
			},
		}));
		queryClient.invalidateQueries({ queryKey: ["channels"] });
	}, [pushMessage]);

	const handleTypingState = useCallback((data: unknown) => {
		const event = data as TypingStateEvent;
		setLiveStates((prev) => ({
			...prev,
			[event.channel_id]: {
				...prev[event.channel_id],
				isTyping: event.is_typing,
				messages: prev[event.channel_id]?.messages ?? [],
				historyLoaded: prev[event.channel_id]?.historyLoaded ?? false,
			},
		}));
	}, []);

	const handlers = useMemo(() => ({
		inbound_message: handleInboundMessage,
		outbound_message: handleOutboundMessage,
		typing_state: handleTypingState,
	}), [handleInboundMessage, handleOutboundMessage, handleTypingState]);

	useEventSource(api.eventsUrl, { handlers });

	return (
		<div className="min-h-screen bg-app">
			{/* Header */}
			<div className="border-b border-app-line bg-app-darkBox/50 px-6 py-4">
				<div className="mx-auto flex max-w-5xl items-center justify-between">
					<div>
						<h1 className="font-plex text-lg font-semibold text-ink">Spacebot</h1>
						<p className="text-tiny text-ink-faint">Control Interface</p>
					</div>
					{statusData && (
						<div className="flex items-center gap-3 text-sm">
							<div className="flex items-center gap-1.5">
								<div className="h-2 w-2 rounded-full bg-green-500" />
								<span className="text-ink-dull">Running</span>
							</div>
							<span className="text-ink-faint">
								{formatUptime(statusData.uptime_seconds)}
							</span>
							<span className="text-ink-faint">
								PID {statusData.pid}
							</span>
						</div>
					)}
				</div>
			</div>

			{/* Content */}
			<div className="mx-auto max-w-5xl p-6">
				<div className="mb-4 flex items-center justify-between">
					<h2 className="font-plex text-sm font-medium text-ink-dull">
						Active Channels
					</h2>
					<span className="text-tiny text-ink-faint">
						{channels.length} channel{channels.length !== 1 ? "s" : ""}
					</span>
				</div>

				{channelsLoading ? (
					<div className="flex items-center gap-2 text-ink-dull">
						<div className="h-2 w-2 animate-pulse rounded-full bg-accent" />
						Loading channels...
					</div>
				) : channels.length === 0 ? (
					<div className="rounded-lg border border-dashed border-app-line p-8 text-center">
						<p className="text-sm text-ink-faint">
							No active channels. Send a message via Discord, Slack, or webhook to get started.
						</p>
					</div>
				) : (
					<div className="grid gap-3 sm:grid-cols-2">
						{channels.map((channel) => (
							<ChannelCard
								key={channel.id}
								channel={channel}
								liveState={liveStates[channel.id]}
							/>
						))}
					</div>
				)}
			</div>
		</div>
	);
}

export function App() {
	return (
		<QueryClientProvider client={queryClient}>
			<Dashboard />
		</QueryClientProvider>
	);
}
