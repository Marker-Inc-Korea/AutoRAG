/**
 * shadcn/ui Switch (new-york-v4) adapted to our token CSS.
 *
 * Same Radix primitive and markup as the registry item (`radix-ui`
 * Root/Thumb, data-state attributes); the registry's Tailwind utilities
 * map onto the matching `--settings-switch` rules in styles/shell.css.
 * Colors follow the v6 handoff tokens, not the shadcn palette.
 */

import type { ReactElement } from "react";
import { Switch as SwitchPrimitive } from "radix-ui";

export function Switch({
	checked,
	onChange,
	disabled = false,
	ariaLabel,
}: {
	readonly checked: boolean;
	readonly onChange: () => void;
	readonly disabled?: boolean | undefined;
	readonly ariaLabel?: string | undefined;
}): ReactElement {
	return (
		<SwitchPrimitive.Root
			className="settings-switch"
			checked={checked}
			disabled={disabled}
			aria-label={ariaLabel}
			onCheckedChange={() => onChange()}
		>
			<SwitchPrimitive.Thumb className="settings-switch__thumb" />
		</SwitchPrimitive.Root>
	);
}
