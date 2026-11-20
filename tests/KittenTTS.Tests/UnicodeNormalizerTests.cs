using System.Text;
using KittenTTS.Text;

namespace KittenTTS.Tests;

/// <summary>The table-driven normalizer (used under InvariantGlobalization) on cases with known answers.</summary>
public sealed class UnicodeNormalizerTests
{
    [Theory]
    [InlineData("café", NormalizationForm.FormD, "café")]
    [InlineData("café", NormalizationForm.FormC, "café")]
    [InlineData("ﬁ½", NormalizationForm.FormKC, "fi1⁄2")]
    [InlineData("Zürich", NormalizationForm.FormKD, "Zürich")]
    [InlineData("한국어", NormalizationForm.FormD, "한국어")]
    [InlineData("한국어", NormalizationForm.FormC, "한국어")]
    [InlineData("q̣̇", NormalizationForm.FormC, "q̣̇")]          // canonical reordering
    [InlineData("ḍ̇", NormalizationForm.FormC, "ḍ̇")]            // d + dot above + dot below
    [InlineData("Ａｂｃ１２３", NormalizationForm.FormKC, "Abc123")]
    public void ManagedMatchesUnicode(string input, NormalizationForm form, string expected) =>
        Assert.Equal(expected, UnicodeNormalizer.Managed(input, form));
}
