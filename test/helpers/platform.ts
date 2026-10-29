/**
 * Runs `run` while `process.platform` reports `platform`, then restores the real value.
 *
 * For code that reads the platform at call time. Datasources bound to one platform (kakao: macOS only) are
 * built only there, so a test that exercises one off its platform builds it inside this wrapper.
 */
export function withPlatform<T>(platform: NodeJS.Platform, run: () => T): T {
	const real = Object.getOwnPropertyDescriptor(process, "platform") as PropertyDescriptor;
	Object.defineProperty(process, "platform", { ...real, value: platform });
	try {
		return run();
	} finally {
		Object.defineProperty(process, "platform", real);
	}
}
