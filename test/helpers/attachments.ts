import { AttachmentRootsSkill } from "../../src/datasource/attachments.ts";
import type { DatasourceSkill } from "../../src/datasource/types.ts";

/** The skill the factory built, without the attachment-folder wrapper it adds when a datasource holds files. */
export function unwrapAttachments(skill: DatasourceSkill | undefined): DatasourceSkill | undefined {
	return skill instanceof AttachmentRootsSkill ? skill.inner : skill;
}
