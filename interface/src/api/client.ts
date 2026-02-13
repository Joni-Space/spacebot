const API_BASE = "/api";

export interface StatusResponse {
	status: string;
	pid: number;
	uptime_seconds: number;
}

export interface ChannelInfo {
	agent_id: string;
	id: string;
	platform: string;
	display_name: string | null;
	is_active: boolean;
	last_activity_at: string;
	created_at: string;
}

export interface ChannelsResponse {
	channels: ChannelInfo[];
}

export interface InboundMessageEvent {
	type: "inbound_message";
	agent_id: string;
	channel_id: string;
	sender_id: string;
	text: string;
}

export interface OutboundMessageEvent {
	type: "outbound_message";
	agent_id: string;
	channel_id: string;
	text: string;
}

export interface TypingStateEvent {
	type: "typing_state";
	agent_id: string;
	channel_id: string;
	is_typing: boolean;
}

export interface ProcessEventWrapper {
	type: "process_event";
	agent_id: string;
	event: {
		type: string;
		[key: string]: unknown;
	};
}

export type ApiEvent =
	| InboundMessageEvent
	| OutboundMessageEvent
	| TypingStateEvent
	| ProcessEventWrapper;

async function fetchJson<T>(path: string): Promise<T> {
	const response = await fetch(`${API_BASE}${path}`);
	if (!response.ok) {
		throw new Error(`API error: ${response.status}`);
	}
	return response.json();
}

export interface ConversationMessage {
	id: string;
	role: "user" | "assistant";
	sender_name: string | null;
	sender_id: string | null;
	content: string;
	created_at: string;
}

export interface MessagesResponse {
	messages: ConversationMessage[];
}

export const api = {
	status: () => fetchJson<StatusResponse>("/status"),
	channels: () => fetchJson<ChannelsResponse>("/channels"),
	channelMessages: (channelId: string, limit = 20) =>
		fetchJson<MessagesResponse>(
			`/channels/messages?channel_id=${encodeURIComponent(channelId)}&limit=${limit}`,
		),
	eventsUrl: `${API_BASE}/events`,
};
