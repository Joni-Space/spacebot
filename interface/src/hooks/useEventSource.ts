import { useEffect, useRef, useCallback } from "react";

type EventHandler = (data: unknown) => void;

interface UseEventSourceOptions {
	/** Map of SSE event types to handlers */
	handlers: Record<string, EventHandler>;
	/** Whether to connect (default true) */
	enabled?: boolean;
}

/**
 * SSE hook with auto-reconnect and typed event handling.
 * Connects to the given URL, parses each event's JSON data,
 * and dispatches to the matching handler by event type.
 */
export function useEventSource(url: string, options: UseEventSourceOptions) {
	const { handlers, enabled = true } = options;
	const handlersRef = useRef(handlers);
	handlersRef.current = handlers;

	const reconnectTimeout = useRef<ReturnType<typeof setTimeout>>();
	const eventSourceRef = useRef<EventSource>();

	const connect = useCallback(() => {
		if (eventSourceRef.current) {
			eventSourceRef.current.close();
		}

		const source = new EventSource(url);
		eventSourceRef.current = source;

		// Register a listener for each event type in handlers
		for (const eventType of Object.keys(handlersRef.current)) {
			source.addEventListener(eventType, (event: MessageEvent) => {
				try {
					const data = JSON.parse(event.data);
					handlersRef.current[eventType]?.(data);
				} catch {
					// non-JSON data, pass raw
					handlersRef.current[eventType]?.(event.data);
				}
			});
		}

		source.onerror = () => {
			source.close();
			// Reconnect after 2 seconds
			reconnectTimeout.current = setTimeout(connect, 2000);
		};
	}, [url]);

	useEffect(() => {
		if (!enabled) return;

		connect();

		return () => {
			if (reconnectTimeout.current) {
				clearTimeout(reconnectTimeout.current);
			}
			eventSourceRef.current?.close();
		};
	}, [connect, enabled]);
}
