import { CampaignSimulator } from "@/components/campaign";
import { Missing, PageHeader } from "@/components/ui";
import { artifact, type Row } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function Campaign() {
  const sim = await artifact<Row[]>("campaign_simulation.csv");
  return (
    <>
      <PageHeader title="Campaign Simulator"
        subtitle="If only part of the audience can be treated, who should get it? Incremental outcomes are estimated on the held-out randomized test population, so they are causal, not correlational." />
      {sim ? <CampaignSimulator rows={sim} /> : <Missing what="Campaign simulation" />}
    </>
  );
}
